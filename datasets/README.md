数据集目录格式：
├── 数据集名称
│   └── VOC2007
│       ├── SegmentationClass
│       │    └── 存放PNG格式的灰度标签
│       ├── ImageSets
│       │    └── Segmentation
│       │          └──这部分通过create_dataset.py生成训练集和验证集
│       ├── JPEGImages
│       │    └── 存放JPG格式的原始图片