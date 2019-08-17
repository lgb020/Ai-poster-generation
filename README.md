# Ai poster generation
 2019deecamp夏令营广州站49组 <神来之笔——自动生成海报> 项目相关资料
![合照](media/%E5%90%88%E7%85%A7.jpeg)

## 2019deecamp分为四周 
第一周为理论实践课程
文件[课程表](media/%E8%AF%BE%E7%A8%8B%E8%A1%A8.pdf)是广州站的，其他地区(北京、南京、上海)课程一样，授课方式不一样。
[思维导图](./media/mindmap)献上，既可查询也可扩展有兴趣的开发者扩展横向知识[导图归档](media/%E5%AF%BC%E5%9B%BE%E5%BD%92%E6%A1%A3.png)

第二周到第四周项目实战
![海报](media/%E6%B5%B7%E6%8A%A5.png)

## 项目介绍
1. WEB端呈现
![网页展示](media/%E7%BD%91%E9%A1%B5%E5%B1%95%E7%A4%BA-1.png)
1. 效果

<img src="media/%E6%95%88%E6%9E%9C1.png" width="500" hegiht="313" align=center />

3. 对比与参考

| 传统生成海报的不足 | 本项目的思考方法 |
| --- | --- |
| 素材固定：由内部限定的素材组合，含义抽象 | 素材多样化 用户自定义上传图片或者文字指定素材 | 
|  风格固定：单个背景替换，只能简单支持更换颜色| 风格多样化 支持对同一张海报进行多风格的转换 | 
|  模板固定：无法自己增加logo或者调整文本框的位置 | 模板多样化 支持多种文字和图片的排版样式 | 
|  | 

| 相关技术 | 相关资料 |
| --- | --- |
| BERT命名实体识别 | [[论文链接]](https://arxiv.org/abs/1810.04805) |
| PoolNet抠图 | [[论文链接]](https://arxiv.org/abs/1904.09569) |
| 风格迁移 | [[论文链接]](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) |
| 超分辨率 | [[论文链接]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf) |
| 规则:模板多样化 | |
| Web端实现 | Flask |

## 使用方法
1. 安装相关环境
    * requirement...
    * 镜像...
    
2. 本地端部署
    * 在[WEB](./WEB)目录下运行 ```flask run```
    * 打开**http://127.0.0.1:5000**网址
3. 服务端部署
    * 打开80端口后在[WEB](./WEB)目录下运行```python app.py```
    * 打开**xxx.xxx.xxx.xxx**公网网址
    
## 模型文件使用
1. 模型下载
    * [Google云盘](https://drive.google.com/drive/folders/1dc12sjI0S7-GLn0208qa1L209qHRC1jD?usp=sharing)
    * [百度云盘](https://pan.baidu.com/s/1IkYpqnj77P27OUHS1Vzu_A)提取码: 4949
1. 模型存放
    * final.pth --> poster_project/SEG/results/run-1/models/
    * model.ckpt-1524.data-00000-of-00001 --> poster_project/NLP/ner-part/bert/output/result_dir/
    * vgg19-dcbb9e9d.pth --> 在Web端运行自定义风格时请查阅命令栏储存地址