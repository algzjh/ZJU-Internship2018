from flask import Flask, jsonify, json
from flask import render_template,request
from flask import url_for,redirect
from flask_cors import CORS
import os
import time
from calcandshow import  *

r = {}
mp = {}
rmp = {}
yearnum = 11
peonum = 429
clusternum = 8;
app = Flask(__name__)

@app.route('/test') #装载到对应的URL地址
def test():
    return 'test'

@app.route('/getData',methods = ['POST'])
def get_Data():
    r = request.get_json()
    mp,rmp = getnamemap(r)
    R,Rv = getRankMatrix()
    return json.dumps({'R':R,'Rv':Rv})

@app.route('/getTSNE',methods = ['POST'])
def get_TSNE():
    R = request.get_json()
    R_tsne = TSNE().fit_transform(R)
    return  json.dumps(R_tsne)


@app.route('/setClusternum',methods = ['POST'])
def set_Clusternum():
    clusternum = request.args.get('clusternum','')
    return 'OK'

@app.route('calcKmeans',methods = ['POST'])
def calc_Kmeans():
    myR_tsne = request.get_json()
    label_pred = calcKmeans(myR_tsne)
    return json.dumps(label_pred)

if __name__ == '__main__':#当本模块被直接启动时才运行
    app.debug = True
    app.run()
