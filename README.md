# Hierarchical-Template-Transformer

This is code for Hierarchical template transformer for fine-grained sentiment controllable generation (https://www.sciencedirect.com/science/article/abs/pii/S0306457322001546)

The datasets (FSCG-80) can be found [this](https://drive.google.com/drive/folders/1lXZLdfkb8hskR5nI9Tqu1uhXC8N1JvbU) and the snippext [this](https://github.com/rit-git/Snippext_public)

Regarding the description of the about_Snippext , we have made certain modifications and additions to Snippext:

1. For the [yelp_data_prepare.py](https://github.com/YuanLi95/Hierarchical_Template_Transformer/tree/main/About_Snippext/yelp_data), it filters the original yelp with  conditions such as the users, sentence max length, etc.  Runing this file, you can get the yelp_aspect_datasets.csv.

2. run_pipeline.py replaces the original Snippext corresponding file, from which the do_pairing method can obtain aspect, opinion, and sentiment polarity. Afther runing this file, you can obtained the /aspect_datasets/datasets.jsonl.

3. The purpose of the prepare.py file is to convert datasets.jsonl into the final dataset (train, dev, test)

If you have any question，you can contact me at ynuyuanli@gmail.com or wechat:a374702739

