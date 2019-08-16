from flask import Flask,render_template,request,send_from_directory
from flask import url_for
from flask import session
from datetime import timedelta
import re
import torch
import json
from werkzeug import secure_filename
import os,sys
import click
import time
import random
import multiprocessing

app=Flask(__name__)
app.config['UPLOAD_PATH'] = '/static/img/'
app.config["SECRET_KEY"] = "renyizifuchuan"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=1)
# app.config['UPLOAD_FOLDER']='/before'


#-------------------- 外部库 -------------------------
import sys
sys.path.append('../NLP/ner-part/')
from ner_main import NER # 自动切换GPU
# from ner_main import * 

sys.path.append('../RELU/')
from template import choosetemplate,addallimage # CPU处理

sys.path.append('../ST/')
from transformer_net import TransformerNet
from imageStyle_transfer import stylize # 无须更改

sys.path.append('../ST/online/')
from style_transfer import onlinestyle
st_use_cuda = torch.cuda.is_available() and True

sys.path.append('../SEG/')
from xjytemp import koutu # 可更改joint_solver L13

sys.path.append('../SR/')
from SR_test import generateSR # 可更改 L30
#-------------------- 外部库 -------------------------


@app.route('/')
@app.route('/home')
@app.route('/home.html')
def home():
	return render_template('home.html')


@app.route('/index')
@app.route('/index.html')
def index_1():
	return render_template('index.html')

@app.route('/information')
@app.route('/information.html')
def information():
	return render_template('information.html')

ner_a = ""
ner_b = ""
ner_c = ""
def ner_pr(name):
	string = open('./test/'+name,'r',encoding='utf-8').read()
	session['title'],session['comment_1'],session['comment_2'] = NER(input_string=string)
	print(session['title'],session['comment_1'],session['comment_2'])
	ner_a,ner_b,ner_c = session['title'],session['comment_1'],session['comment_2']
@app.route('/upload',methods=['GET', 'POST'])
def upload_file():
	if request.method=='POST':
		session['title'] = ""
		session['comment_1'] = ""
		session['comment_2'] = ""
		f=request.files['file']
		f.save('./test/'+f.filename)
		# p = multiprocessing.Process(target=ner_pr, args=(f.filename,))
		# p.start()
		# p.join()
		
		string = open('./test/'+f.filename,'r',encoding='utf-8').read()
		session['title'],session['comment_1'],session['comment_2'] = NER(input_string=string)
		# session['title'],session['comment_1'],session['comment_2'] = ner_a,ner_b,ner_c
		# print(session['title'],session['comment_1'],session['comment_2'])
		return render_template('analysis.html',title=session['title'],comment_2=session['comment_2'],comment_1=session['comment_1'])

@app.route('/loadinformation',methods=['POST'])
def loadinformation():
	title=request.form.get('title')
	comment_1=request.form.get('comment_1')
	comment_2=request.form.get('comment_2')
	if title == "":
		title = "第十八届上海国际汽车工业展览会"
	if comment_1 == "":
		comment_1 = "4月16日"
	if comment_2 == "":
		comment_2 = "上海国际会展中心"
	session['title']=title
	session['comment_1']=comment_1
	session['comment_2']=comment_2
	return render_template('analysis.html',title=title,comment_2=comment_2,comment_1=comment_1)

@app.route("/analysis")
@app.route("/analysis.html")
def analysis():
	return render_template('analysis.html',title=session['title'],comment_2=session['comment_2'],comment_1=session['comment_1'])

@app.route("/select")
@app.route("/select.html")
def select():
	val1=time.time()
	session["imagepath_1"]="img/foreground/benz-001.png"
	session["imagepath_2"]="img/background/IMG_0797.JPG"
	return render_template("select.html",imagepath_1=session["imagepath_1"],imagepath_2=session["imagepath_2"],val1=val1)



@app.route('/upload_select1',methods=['GET', 'POST'])
def upload_file1():
	if request.method=='POST':
		f=request.files['file']
		f.save('./static/img/'+f.filename)

		session["imagepath_1"]="img/"+f.filename
		val1=time.time()

		
		koutu('./static/' + session["imagepath_1"], './static/img/koutu.png')
		torch.cuda.empty_cache()
		session["imagepath_1"] = 'img/koutu.png'
		return render_template('select.html',imagepath_1= session["imagepath_1"],imagepath_2=session["imagepath_2"],val1=val1)

@app.route('/upload_select2',methods=['GET', 'POST'])
def upload_file2():
	if request.method=='POST':
		f=request.files['file']
		f.save('./static/img/'+f.filename)
		session["imagepath_2"]="img/"+f.filename
		val1=time.time()
		return render_template('select.html',imagepath_1=session["imagepath_1"],imagepath_2=session["imagepath_2"],val1=time.time())

@app.route("/select_random1")
def select_random1():
	file_list=os.listdir('./static/img/foreground/')
	rand=random.randint(0,len(file_list)-1)
	session["imagepath_1"]="img/foreground/"+file_list[rand]
	val1=time.time()
	return render_template('select.html',imagepath_1=session["imagepath_1"],imagepath_2=session["imagepath_2"],val1=time.time())

@app.route("/select_random2")
def select_random2():
	file_list=os.listdir('./static/img/background/')
	rand=random.randint(0,len(file_list)-1)
	session["imagepath_2"]="img/background/"+file_list[rand]
	val1=time.time()
	return render_template('select.html',imagepath_1=session["imagepath_1"],imagepath_2=session["imagepath_2"],val1=time.time())

@app.route("/style",methods=['GET', 'POST'])
@app.route("/style.html",methods=['GET', 'POST'])
def style():
	session['style_img']="img/field.jpg"
	return render_template("style.html",style_img=session['style_img'])

@app.route("/style_data",methods=['GET', 'POST'])
def style_data():
	style=request.form.get('style')
	modelname = {'X':'hiphop.pth',
				'A':'rain_princess.pth',
				'B':'starry-night.model',
				'C':'style6.pth',
				'D':'style8.pth',
				'E':'style9.pth'}
	txt = []
	txt.append(session['title'])
	txt.append(session['comment_1']+'\n'+session['comment_2'])

	pos = choosetemplate('./static/' + session["imagepath_2"],'./static/' + session["imagepath_1"],
												txt,
												"./static/img/"+style+"results_notext.jpg")
	if style != 'Z':
		device = torch.device("cuda")
		with torch.no_grad():
			style_model = TransformerNet()
			state_dict = torch.load('../ST/saved_models/'+modelname[style])
			# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
			for k in list(state_dict.keys()):
				if re.search(r'in\d+\.running_(mean|var)$', k):
					del state_dict[k]
			style_model.load_state_dict(state_dict)
			style_model.to(device)
		stylize("./static/img/"+style+"results_notext.jpg", "./static/img/"+style+"results_notext_S.png", model=style_model,device=device)
		torch.cuda.empty_cache()
		# stylize("./static/img/"+style+"results_notext.jpg", "./static/img/"+style+"results_notext_S.png", model='../ST/saved_models/'+modelname[style])
		addallimage("./static/img/"+style+"results_notext_S.png",pos,"./static/img/results.jpg")
	else:
		addallimage("./static/img/"+style+"results_notext.jpg",pos,"./static/img/results.jpg")
	torch.cuda.empty_cache()
	return render_template("style.html",style_img=session['style_img'])




@app.route('/upload_style',methods=['GET', 'POST'])
def upload_file3():
	print('session["style_img"]')

	if request.method=='POST':
		f=request.files['file']
		f.save('./static/img/'+f.filename)
		session["style_img"]="img/"+f.filename

		txt = []
		txt.append(session['title'])
		txt.append(session['comment_1']+'\n'+session['comment_2'])
	
		pos = choosetemplate('./static/' + session["imagepath_2"],'./static/' + session["imagepath_1"],
													txt,
													"./static/img/"+"zdy"+"results_notext.jpg")
		addallimage("./static/img/"+"zdy"+"results_notext.jpg",pos,"./static/img/results_h.jpg")
		
		onlinestyle("./static/img/results_h.jpg", "./static/" + session["style_img"], "./static/img/results.jpg", 20,st_use_cuda)
		torch.cuda.empty_cache()
		return render_template("style.html",style_img=session['style_img'])

@app.route('/select_SR',methods=['GET', 'POST'])
def SR_fun():
	multiple=request.form.get('multiple')
	print(type(multiple),multiple)
	generateSR("./static/img/results.jpg","./static/img/results.jpg",int(multiple))
	torch.cuda.empty_cache()
	return render_template("finish.html",finish_img="img/results.jpg")

@app.route("/finish",methods=['GET', 'POST'])
@app.route("/finish.html",methods=['GET', 'POST'])
def finish():
	session['finish_img']="img/results.png"
	return render_template("finish.html",finish_img=session['finish_img'])

@app.route('/download/<path:filename>')
def send_img(filename):
	print(filename.split("/")[1])
	return send_from_directory('static/img/', filename.split("/")[1], as_attachment=True)


@app.errorhandler(404)
def page_not_found(e):
	return render_template('404.html'), 404
	
if __name__ == '__main__':
	app.jinja_env.auto_reload = True
	app.run(host="0.0.0.0",port=80,debug =True)
