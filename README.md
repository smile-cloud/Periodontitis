# 3D牙周炎项目README

3D的牙周炎项目主要包括三个部分：**3D牙齿的实例分割**、**牙槽骨的语义分割**、**传统视觉对分割结果进行处理和分级**。下面的内容介绍也是大体按这个顺序来的。

## 1.3D牙齿的实例分割

3D牙齿本项目采取了两种实现方式，大体思路都差不多，最关键部分都是如何找到**每个牙齿的质心**，从而可以根据质心来切出单颗牙齿的区域，然后进行单个牙齿的分割，最后将所有单颗牙齿进行汇总，从而形成一个nii.gz文件的牙齿实例分割内容。

### 1.1 cui的实例分割思路

cui的论文[在这](https://www.nature.com/articles/s41467-022-29637-2)，主要的思路是先找到**ROI**，在通过神奇的**投票方式找到每颗牙齿的质心**，最后进行**单颗牙齿的分割**这三个步骤。具体对应代码为Tooth-and-alveolar-bone-segmentation-from-CBCT-main目录下的**roi_localization/**、**tooth_detection/**、**single_tooth_segmentation**/，对应的代码和数据可以参考[github](https://github.com/1DreamCollector/Tooth-and-alveolar-bone-segmentation-from-CBCT-main)的实现，也可以参考我百度网盘的实现。百度网盘里面是和数据一起的，可能有点大。 https://pan.baidu.com/s/1Y28b3rqZ8792waOEJjbpYg?pwd=pjs2 

### 1.2基于2D追踪的实例分割思路

这个的思路是通过从Z轴自上而下的牙齿的分割，以及相邻两个切片之间IOU的计算，来把牙齿给抠出来。但是这种出来的牙齿层级感非常严重，表面不平滑。用这种方法的原因是希望通过这个粗颗粒度的分割，将牙齿的质心给找出来，然后再去进行一个单牙的分割。下面是运行步骤。代码在yolo文件夹下：

1. 我无比建议你从源码安装[ultralytics](https://docs.ultralytics.com/quickstart/)，安装到yolo目录下
2. cbctZ轴跟踪_train.ipynb是训练的代码，cbctZ轴追踪_predict.ipynb是测试的代码，**cbctZ轴追踪/**是保存的过程文件，yaml是配置文件
3. 你修改了需要的路径之后就可以运行代码了
4. 如果你想要这个训练的dataset请私聊我，我感觉重新训练不是很有必要。

以上是牙齿实例分割的思路。但是在这泼个凉水，以上的代码训练出来的效果，即使在过拟合的情况下，精度也达不到后面对牙周炎判定的要求。虽然上面的效果已经不错了，针对3D的dice值等一系列的指标已经是SOTA了。但是，后面是要对单颗牙进行牙冠的取点的，只要牙齿的预测有一点的偏差，都会导致牙冠的点提取不出来。所以上面的第二种方法是**方法上**的创新点，但不是**实际**的落地点和使用点。后面涉及到**牙冠提取**和**牙根提取**这两个单牙的部分，是直接使用的标注文件，即使是这样，还是有很多牙冠和牙根提取不准确，等你做后续步骤遇到问题的时候可以再交流。

## 2.牙槽骨的语义分割

这部分是用Paddle的nnunet做的，代码在paddle文件夹下：

1. 首先还是无比建议从源码构建[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/docs/install_cn.md)
2. 然后见第四部分

## 3.传统视觉对分割结果进行处理和分级

传统视觉的部分都在run文件夹下，我已经尽力整理了代码并控制和缩小了大小。具体解释如下：

1. 得到了牙齿的分割结果和牙槽骨的分割结果之后，使用run/postprocess_pred.ipynb进行合并，注意nnunet预测出来的牙槽骨是很乱的，需要先提取最大连通域
2. 然后针对每一颗牙，运行run/提取牙冠和对应牙槽骨距离.ipynb，这个是寻找指定位置的切片的。

## 4.说明

有一部分代码因为autodl一段时间没开机被删了，要自己写hhh。

1. 针对单个切片上提取**牙槽骨**、**牙冠**、**牙根**的点。以下最好是基于**标注文件**来做

   1. **牙槽骨**：上牙计算牙槽骨和牙接触的最低点，下牙计算牙槽骨和牙接触的最高点。
   2. **牙根**：最高点往下做1/4牙齿高度的线，先判断几个牙根点
   3. **牙冠**：上牙从下往上，下牙从上往下，找到第一个颜色突变点

2. 牙槽骨的nnunet训练

   1. 适配的PaddleSeg的数据集已经做好了，只需要运行训练命令就行。数据集下载：https://pan.baidu.com/s/1uazRO36LStoOq3h8N6guhw?pwd=9s2e
   2. 需要将yaml文件放到configs/nnunet中
   3. 下面的流程可以只进行训练和预测。主要，预测是要把要预测的文件放到**test**文件目录下，并且名字要以_0000.nii.gz结尾。postprocessing_json_path对应的postprocessing.json可以为空，可以使用第三部分run/postprocess_pred.ipynb中的提取连通域来做后处理
   4. 训练的其他信息可以参考[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10)

   ```shell
   # 训练
   python train.py --config configs/nnunet/xiahe_mri/fullres_xiahe_mri.yaml --log_iters 20 --precision fp16 --nnunet --save_dir output/xiahe_mri --save_interval 200 --use_vdl --do_eval
   # 预测：
   python nnunet/predict.py --image_folder test  --output_folder predict/ --plan_path data/xiahe_mri/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/xiahe_mri/best_model/model.pdparams --postprocessing_json_path data/xiahe_mri/postprocessing.json --model_type cascade_lowres 
   # 模型导出：
   python nnunet/export.py --config configs/nnunet/yahegu/nnunet_3d_fullres_yahegu_fold0.yml --save_dir output/static/3d_unet/fold0 --model_path output/3d_unet/fold0/best_model/model.pdparams
   # infer:
   python nnunet/infer.py --image_folder test  --output_folder output/nnunet_static/3d_fullres --plan_path yahegu/yahegu_dataset/nnUNetPlansv2.1_plans_3D.pkl --model_paths output/static/3d_unet/fold0/model.pdmodel --param_paths output/static/3d_unet/fold0/model.pdiparams --postprocessing_json_path output/3d_unet_val/postprocessing.json --model_type 3d --disable_postprocessing  --save_npz
   ```

   ## 5.建议

   1. 多可视化，多可视化，多可视化！(推荐使用ITK-SNAP打开nii.gz文件)
   2. 有什么问题可以微信交流 vx:18839549069

