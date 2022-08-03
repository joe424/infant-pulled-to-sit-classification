#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cgi
import sys
import time
import glob
import cgitb
import shutil
import subprocess
from subprocess import Popen, PIPE, STDOUT


UPLOAD_DIR = '../2D_pose_estimation/videos/pull_to_sit'

def save_uploaded_file():
    print('''Content-type: text/html

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>inference</title>
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css"
      integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2"
      crossorigin="anonymous"
    />
    <link
      href="https://fonts.googleapis.com/css?family=Source+Code+Pro"
      rel="stylesheet"
    />
    <link
      rel="icon"
      type="image/png"
      href="https://cdn0.iconfinder.com/data/icons/small-n-flat/24/678068-terminal-512.png"
    />
    <style>
      * {
          font-family: 'Source Code Pro', monospace;
          font-size: 1rem !important;
      }
      body {
          background-color: #212529;
      }
      pre {
          color: #cccccc;
      }
      b {
          color: #01b468;
      }
    </style>
  </head>
  <body>
    <table class="table table-dark table-bordered">
      <thead>
        <tr>
          <th scope="col">terminal</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><pre id="s1" class="mb-0"></pre></td>
        </tr>
      </tbody>
    </table>
  </body>
</html>
    ''', flush=True)

    form = cgi.FieldStorage()

    # if not form.__contains__('file'):
    #     print('<script>document.getElementById("s1").innerHTML += "Not found parameter: file";</script>', flush=True)
    #     return
    # form_file = form['file']
    # if not form_file.file:
    #     print('<script>document.getElementById("s1").innerHTML += "Not found parameter: file";</script>', flush=True)
    #     return
    # if not form_file.filename:
    #     print('<script>document.getElementById("s1").innerHTML += "Not found parameter: filename";</script>', flush=True)
    #     return

    if 'file' in form:
      form_file = form['file']
#       os.system('rm ./infant_diagnosis/videos/prone/*.mp4')
#       os.system('rm ./infant_diagnosis/videos/pull_to_sit/*.mp4')
#       os.system('rm ./infant_diagnosis/videos/supine/*.mp4')
      if not isinstance(form_file , list):
        form_file = [form_file]
#       for fileitem in form_file:
#         if fileitem.filename:
#           if re.search("^\d+_\d+m_Prone_\d+\.mp4$", os.path.basename(fileitem.filename)) == None and \
#              re.search("^\d+_\d+m_Pull_to_sit_\d+\.mp4$", os.path.basename(fileitem.filename)) == None and \
#              re.search("^\d+_\d+m_Supine_\d+\.mp4$", os.path.basename(fileitem.filename)) == None:
#              print('<script>document.getElementById("s1").innerHTML += "['+os.path.basename(fileitem.filename)+'] File name ERROR";</script>', flush=True)
#              return
      for fileitem in form_file:
        if fileitem.filename:
#           if re.search("^\d+_\d+m_Prone_\d+\.mp4$", os.path.basename(fileitem.filename)) != None:
#             uploaded_file_path = os.path.join(UPLOAD_DIR, 'prone', os.path.basename(fileitem.filename))
#           elif re.search("^\d+_\d+m_Pull_to_sit_\d+\.mp4$", os.path.basename(fileitem.filename)) != None:
#             uploaded_file_path = os.path.join(UPLOAD_DIR, 'pull_to_sit', os.path.basename(fileitem.filename))
#           elif re.search("^\d+_\d+m_Supine_\d+\.mp4$", os.path.basename(fileitem.filename)) != None:
#             uploaded_file_path = os.path.join(UPLOAD_DIR, 'supine', os.path.basename(fileitem.filename))
          uploaded_file_path = os.path.join(UPLOAD_DIR, os.path.basename(fileitem.filename))
          with open(uploaded_file_path, 'wb') as fout:
            shutil.copyfileobj(fileitem.file, fout)
            # while True:
            #     chunk = form_file.file.read(100000)
            #     if not chunk:
            #         break
            #     fout.write (chunk)
          os.chmod(uploaded_file_path, 0o600)

    # if re.search("^\d+_\d+m_Prone_\d+\.mp4$", os.path.basename(form_file.filename)) != None:
    #   uploaded_file_path = os.path.join(UPLOAD_DIR, 'prone', os.path.basename(form_file.filename))
    # elif re.search("^\d+_\d+m_Pull_to_sit_\d+\.mp4$", os.path.basename(form_file.filename)) != None:
    #   uploaded_file_path = os.path.join(UPLOAD_DIR, 'pull_to_sit', os.path.basename(form_file.filename))
    # elif re.search("^\d+_\d+m_Supine_\d+\.mp4$", os.path.basename(form_file.filename)) != None:
    #   uploaded_file_path = os.path.join(UPLOAD_DIR, 'supine', os.path.basename(form_file.filename))
    # else:
    #   print('<script>document.getElementById("s1").innerHTML += "File name ERROR";</script>', flush=True)
    #   return

    # # uploaded_file_path = os.path.join(UPLOAD_DIR, os.path.basename(form_file.filename))
    # with open(uploaded_file_path, 'wb') as fout:
    #     while True:
    #         chunk = form_file.file.read(100000)
    #         if not chunk:
    #             break
    #         fout.write (chunk)
    # os.chmod(uploaded_file_path, 0o600)

    os.chdir("../")
    #                                                                                 <env_name>
    process = subprocess.Popen('/opt/conda/condabin/conda run --no-capture-output -n EvoSkeleton python -u run.py'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''): 
      if c.decode("utf-8") == '\n':
        print('<script>document.getElementById("s1").innerHTML += "&#13;&#10;";</script>', flush=True)
      else:
        print('<script>document.getElementById("s1").innerHTML += "'+ c.decode("utf-8") +'";</script>', flush=True)
        
if __name__ == '__main__':
    cgitb.enable()
    save_uploaded_file()