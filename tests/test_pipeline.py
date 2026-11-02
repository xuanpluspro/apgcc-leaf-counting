import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import checkpointing
import train
from check_data import audit
from counting_metrics import PointMetrics, match_counts
from dataset import RiceW3999, collate, read_points
from engine import select_points
from infer import tile_axis
from losses import PointCriterion
from models import APGCC, ModelConfig
from models.decoder import AnchorPoints
from models.modules import ifi_simfpn, make_coord
from runtime import resolve_device

torch.set_num_threads(2)


def small_config(**kwargs):
    return ModelConfig(inner_planes=8, head_layers=(16,16), pos_dim=8, **kwargs)


def make_dataset(root):
    for folder in ('JPEGImages','PointAnnotations','ImageSets'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    names = ['wheat1799__中文 name', 'ricetip__empty', 'wheat500__val']
    for index,name in enumerate(names):
        image=np.zeros((64,64,3),dtype=np.uint8)
        image[...,index]=np.arange(64,dtype=np.uint8)[None,:]*3
        Image.fromarray(image).save(root/'JPEGImages'/f'{name}.PNG')
        (root/'PointAnnotations'/f'{name}.txt').write_text('' if index==1 else '20 21\n31 33\n')
    (root/'ImageSets/train.txt').write_text('\ufeff'+names[0]+'\n'+names[1]+'\n')
    (root/'ImageSets/val.txt').write_text(names[2]+'\n')


class DataTest(unittest.TestCase):
    def test_spaces_unicode_empty_points_and_augmented_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_dataset(root)
            dataset=RiceW3999(root,'train',32)
            for index in range(2):
                image,target=dataset[index]
                self.assertEqual(tuple(image.shape),(3,32,32))
                self.assertEqual(target['points'].shape[1],2)
                self.assertTrue(((target['points']>=0)&(target['points']<32)).all())
            self.assertEqual(len(dataset[1][1]['points']),0)
            self.assertEqual(audit(root,True)['train']['images'],2)

    def test_original_sizes_and_padding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_dataset(root)
            path=root/'JPEGImages/wheat500__val.PNG'
            Image.new('RGB',(513,1023)).save(path)
            dataset=RiceW3999(root,'val',augment=False)
            image,target=dataset[0]
            self.assertEqual(tuple(image.shape),(3,1024,528))
            self.assertEqual(target['size'],(1023,513))
            torch.testing.assert_close(target['points'],torch.tensor([[20.,21.],[31.,33.]]))

    def test_invalid_points_duplicate_ids_and_missing_pairs_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_dataset(root)
            path=root/'PointAnnotations/wheat500__val.txt'
            path.write_text('64 1\n')
            with self.assertRaisesRegex(ValueError,'outside'):
                RiceW3999(root,'val',augment=False)[0]
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                RiceW3999(root,'val')
            (root/'ImageSets/train.txt').write_text('ricetip__empty\nricetip__empty\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                RiceW3999(root)

    def test_crop_and_flip_transform_xy_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_dataset(root)
            dataset=RiceW3999(root,'train',32)
            with patch('dataset.random.uniform',return_value=1.), \
                 patch('dataset.random.randint',side_effect=[10,20]), \
                 patch('dataset.random.random',return_value=1.):
                image,target=dataset[0]
            torch.testing.assert_close(target['points'],torch.tensor([[10.,1.],[21.,13.]]))
            with patch('dataset.random.uniform',return_value=1.), \
                 patch('dataset.random.randint',side_effect=[0,0]), \
                 patch('dataset.random.random',return_value=0.):
                image,target=dataset[0]
            torch.testing.assert_close(target['points'],torch.tensor([[11.,21.]]))

    def test_512_and_1024_sources_collate_into_same_training_batch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_dataset(root)
            for name,size in [('wheat1799__中文 name',1024),('ricetip__empty',512)]:
                Image.new('RGB',(size,size)).save(root/'JPEGImages'/f'{name}.PNG')
            dataset=RiceW3999(root,'train',512)
            images,targets=collate([dataset[0],dataset[1]])
            self.assertEqual(tuple(images.shape),(2,3,512,512))


class ModelTest(unittest.TestCase):
    def test_query_ifi_equals_regular_grid_and_uses_all_corners(self):
        torch.manual_seed(1)
        module=ifi_simfpn(ultra_pe=True,pos_dim=8,local=True,feat_num=1,feat_dim=2,
                          head_layers=[8],require_grad=True)
        feature=torch.arange(2*4*6,dtype=torch.float32).reshape(1,2,4,6).requires_grad_()
        expected=module(feature,size=(3,5),level=1)
        actual=module.query(feature,make_coord((3,5)).unsqueeze(0),1,(3,5))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
        expected.sum().backward()
        self.assertGreater(int((feature.grad!=0).sum()),2)

    def test_train_auxiliary_gradients_and_empty_image(self):
        torch.manual_seed(2)
        model=APGCC(small_config())
        images=torch.randn(2,3,64,64)
        targets=[dict(points=torch.tensor([[20.,20.],[40.,42.]])),dict(points=torch.empty(0,2))]
        outputs=model(images,targets)
        self.assertEqual(tuple(outputs['pred_points'].shape),(2,256,2))
        self.assertEqual(len(outputs['aux']),4)
        losses=PointCriterion()(outputs,targets)
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        losses['loss_aux'].backward()
        for parameter in (model.encoder.body1[0].weight,
                          model.decoder.ifi.offset_head[0].weight,
                          model.decoder.ifi.confidence_head[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(),0)
        model.eval()
        with torch.inference_mode():
            self.assertNotIn('aux',model(images))

    def test_all_empty_batch_backward(self):
        model=APGCC(small_config())
        targets=[dict(points=torch.empty(0,2)) for _ in range(2)]
        output=model(torch.randn(2,3,32,32),targets)
        losses=PointCriterion()(output,targets)
        self.assertEqual(losses['loss_points'].item(),0.)
        self.assertEqual(losses['loss_aux'].item(),0.)
        losses['loss'].backward()

    def test_auxiliary_queries_at_regular_anchors_use_identical_prediction_path(self):
        model=APGCC(small_config()).eval()
        model.decoder.aux_number=(1,0)
        model.decoder.aux_range=(0.,8.)
        images=torch.randn(1,3,64,64)
        points=torch.tensor([[2.,2.],[6.,2.],[2.,6.],[6.,6.]])
        with torch.inference_mode():
            features=model.encoder(images)
            maps=model.decoder.encode_features(features)
            regular=model.decoder(images,features)
            auxiliary=model.decoder.auxiliary(images,maps,[dict(points=points)],(8,8))[0]
        torch.testing.assert_close(auxiliary['pred_points'],regular['pred_points'][0,:4],atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(auxiliary['pred_logits'],regular['pred_logits'][0,:4],atol=1e-4,rtol=1e-4)

    def test_mixed_512_1024_full_image_forward(self):
        model=APGCC(small_config()).eval()
        with torch.inference_mode():
            for size in (512,1024):
                result=model(torch.zeros(1,3,size,size))
                self.assertEqual(tuple(result['pred_points'].shape),(1,(size//8)**2*4,2))
                self.assertTrue(torch.isfinite(result['pred_points']).all())

    def test_reference_points_xy_and_strict_shape_validation(self):
        anchor=AnchorPoints(pyramid_levels=3,row=2,line=2)(torch.zeros(1,3,32,32))
        torch.testing.assert_close(anchor[0,:4],torch.tensor([[2.,2.],[6.,2.],[2.,6.],[6.,6.]]))
        with self.assertRaisesRegex(ValueError,'multiples'):
            APGCC(small_config())(torch.zeros(1,3,33,32))

    def test_hungarian_assignment_is_per_image_and_distance_sensitive(self):
        outputs=dict(pred_points=torch.tensor([[[5.,5.],[100.,100.]],[[100.,100.],[5.,5.]]]),
                     pred_logits=torch.zeros(2,2,2))
        targets=[dict(points=torch.tensor([[5.,5.]])) for _ in range(2)]
        match=PointCriterion().match(outputs,targets)
        self.assertEqual(match[0][0].tolist(),[0])
        self.assertEqual(match[1][0].tolist(),[1])


class EvaluationTest(unittest.TestCase):
    def test_cuda_default_resolves_to_explicit_index_before_set_device(self):
        with patch('runtime.torch.cuda.is_available',return_value=True), \
             patch('runtime.torch.cuda.current_device',return_value=0), \
             patch('runtime.torch.cuda.set_device') as setter:
            self.assertEqual(str(resolve_device('cuda')),'cuda:0')
            setter.assert_called_once_with(torch.device('cuda:0'))

    def test_one_to_one_metrics_duplicates_and_empty_images(self):
        self.assertEqual(match_counts([[0,0],[0,0]],[[0,0]],4),(1,1,0))
        meter=PointMetrics()
        meter.update('one',[[0,0],[0,0]],[[0,0]])
        meter.update('empty',[],[])
        result=meter.summary()
        self.assertEqual(result['mae'],.5)
        self.assertAlmostEqual(result['f1_8px'],2/3)

    def test_filter_padding_outside_and_probability(self):
        output=dict(pred_points=torch.tensor([[[1.,1.],[64.,2.],[-1.,2.],[10.,10.]]]),
                    pred_logits=torch.tensor([[[0.,2.],[0.,2.],[0.,2.],[2.,0.]]]))
        points,scores=select_points(output,0,(64,64))
        self.assertEqual(points.tolist(),[[1.,1.]])

    def test_tile_ownership_covers_every_pixel_once(self):
        for length in (512,1024,3072,3101):
            windows=tile_axis(length,1024,64)
            coverage=np.zeros(length,dtype=int)
            for start,end,left,right in windows:
                self.assertGreaterEqual(left,start)
                self.assertLessEqual(right,end)
                coverage+=(np.arange(length)>=left)&(np.arange(length)<right)
            self.assertTrue((coverage==1).all())


class ResumeTest(unittest.TestCase):
    def test_atomic_failure_preserves_old_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'checkpoint.pth'
            checkpointing.atomic_save({'old':1},path)
            before=path.read_bytes()
            def fail(state,stream):
                stream.write(b'partial');raise OSError('disk full')
            with patch('checkpointing.torch.save',side_effect=fail):
                with self.assertRaises(OSError):
                    checkpointing.atomic_save({'new':2},path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(td).glob('*.tmp')),[])

    def test_real_training_resume_equals_uninterrupted_and_exports(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';make_dataset(data)
            common=['--data-path',str(data),'--scratch','--device','cpu','--workers','0',
                    '--epochs','2','--patch-size','32','--batch-size','2','--eval-every','1',
                    '--output-root',str(root/'exp')]
            with patch('train.ModelConfig',side_effect=small_config):
                train.main(train.parser().parse_args(common+['--tag','continuous']))
                original_save=checkpointing.save
                def interrupt_after_checkpoint(path,*args,**kwargs):
                    original_save(path,*args,**kwargs)
                    if Path(path).name=='checkpoint.pth' and args[4]==0:
                        raise KeyboardInterrupt('simulated shutdown after epoch 1')
                with patch('checkpointing.save',side_effect=interrupt_after_checkpoint):
                    with self.assertRaises(KeyboardInterrupt):
                        train.main(train.parser().parse_args(common+['--tag','resumed']))
                train.main(train.parser().parse_args(common+['--tag','resumed']))
            continuous=checkpointing.load(root/'exp/continuous/output/checkpoint.pth')
            resumed=checkpointing.load(root/'exp/resumed/output/checkpoint.pth')
            self.assertEqual(resumed['epoch'],1)
            self.assertEqual(continuous['lr_scheduler'],resumed['lr_scheduler'])
            for key,value in continuous['model'].items():
                torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
            output=root/'exp/resumed/output'
            for file in ('metrics_history.csv','train_history.csv','best_metrics.json',
                         'evaluation/epoch_0002/per_source.csv','evaluation/epoch_0002/per_image.csv'):
                self.assertTrue((output/file).is_file(),file)
            self.assertEqual(len((output/'metrics_history.csv').read_text().splitlines()),3)

    def test_missing_state_and_changed_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args=train.parser().parse_args(['--scratch','--device','cpu'])
            model=APGCC(small_config())
            optimizer=torch.optim.Adam(model.parameters())
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,2)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            path=Path(td)/'checkpoint.pth'
            args.eval_every=5
            checkpointing.save(path,model,optimizer,scheduler,scaler,0,args,3.,0,{'train':'same'})
            state=checkpointing.load(path)
            args.eval_every=1
            result=checkpointing.restore(state,model,optimizer,scheduler,scaler,args,
                                         {'train':'same'},train.logging.getLogger('test'))
            self.assertEqual(result,(1,3.,0))
            args.epochs+=1
            with self.assertRaisesRegex(ValueError,'epochs'):
                checkpointing.restore(state,model,optimizer,scheduler,scaler,args,{'train':'same'},train.logging.getLogger('test'))
            del state['optimizer']
            with self.assertRaisesRegex(ValueError,'missing optimizer'):
                checkpointing.restore(state,model,optimizer,scheduler,scaler,args,{'train':'same'},train.logging.getLogger('test'))


if __name__=='__main__':
    unittest.main()
