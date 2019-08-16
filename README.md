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
最终效果以web端呈现给用户
![网页展示](media/%E7%BD%91%E9%A1%B5%E5%B1%95%E7%A4%BA-1.png)
![效果1](media/%E6%95%88%E6%9E%9C1.png)


| 传统生成海报的不足 | 本项目的思考方法 |
| --- | --- |
| 素材固定：由内部限定的素材组合，含义抽象 | 素材多样化 用户自定义上传图片或者文字指定素材 | 
|  风格固定：单个背景替换，只能简单支持更换颜色| 风格多样化 支持对同一张海报进行多风格的转换 | 
|  模板固定：无法自己增加logo或者调整文本框的位置 | 模板多样化 支持多种文字和图片的排版样式 | 
|  | 

| 相关技术 | 相关资料 |
| --- | --- |
| BERT命名实体识别\PoolNet抠图 | |
| 风格迁移 | |
| 超分辨率 | |
| 规则:模板多样化 | |
| Web端实现 | |

# 使用方法
1. 安装相关环境
2. 在[WEB](./media/mindmap)目录下运行 
```flask run
```
3.打开**http://127.0.0.1:5000/**网址

# 模型文件存放
final.pth --> poster_project/SEG/results/run-1/models/
model.ckpt-1524.data-00000-of-00001
poster_project/NLP/ner-part/bert/output/result_dir/
