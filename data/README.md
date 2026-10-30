将 RiceW3999.tar 放入本目录后解压，得到 data/RiceW3999/。

目录内包含 JPEGImages/、PointAnnotations/、ImageSets/train.txt 和 ImageSets/val.txt。
每行是完整图片 ID（不带扩展名），文件名可以含空格、中文。不要用 split() 拆分清单。
点 TXT 每行严格为 x y，单位为原图像素；空 TXT 表示零个点。可选 test.txt 用于独立测试。
训练接受 512/1024 混合图片，在线裁剪为512；验证保留尺寸。真实数据不会提交到 GitHub。
