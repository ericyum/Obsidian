1. 파일이 처음에 없으므로 "w" 빈파일을 만든다. open
test.txt를 만들고 여기에
"hello world"문자열을 추가
파일을 close하고 종료

2. 파일의 내용을 추가 open("a")
"hello world2"

3. 작성한 파일을 읽기 -> readline()과 readlines()로 한 줄 씩 읽을 수도 있고 여러 줄을 읽을 수도 있음


# test.txt파일을 쓰기모드로 열고 그 안에 "hello world" 문자열을 쓴다.

f = open("test.txt","w")

writeData = "hello world\n"

f.write(writeData)

f.close()

# 위의 셀을 실행하고 이 셀 실행시 덮어쓰기

# open시 'w'는 기존에 파일이 없으면 생성이지만

# 파일이 존재하면 덮어쓰기

f = open("test.txt","a")

writeData = "hello world2"

f.write(writeData)

f.close()



f = open("test.txt", 'r')

while True:

  line = f.readline()

  if not line: break

  print(line, end='')

f.close()




f = open("test.txt", 'r')

lines = f.readlines()

  

print(type(lines))

  

for line in lines:

  print(line, end='')

f.close()



vscode python 뿐만 아니라 다른 여러 언어들로 이루어진 파일들을 종합 적으로 다룰 수 있는 만능 작업대
-> python 확장을 설치하자. 그래야 python을 다룰 때 vscode가 각종 기능들을 지원한다.

anaconda -> 각 파이썬 파일 별 버전 관리(가상 환경을 이용함)


anaconda의 작은 버전인 miniconda 설치 후

anaconda prompt를 실행하면
(base) C:\Users\SBA>
이렇게 뜬다.

1. 가상환경 만들기
(base) C:\Users\SBA>conda create -n pyqt5 python=3.10

2. 가상환경 진입
(pyqt5) C:\Users\SBA>conda activate pyqt5

3. 가상환경에서 나가기
(pyqt5) C:\Users\SBA>conda deactivate

4. 설치되어있는 기능들 출력
(pyqt5) C:\Users\SBA>pip list
Package    Version
---------- -------
pip        25.1
setuptools 78.1.1
wheel      0.45.1

5. 보면 pyqt5 모듈이 설치되지 않은 상태임을 알 수 있다.
(pyqt5) C:\Users\SBA\github\pyqt5_example>pip install pyqt5
따라서 설치를 한다.


6. 새로운 가상환경을 만든 다음에는
vs code에서 ctrl + shift + P를 눌러서 command pallet 창을 나타나게 한 후 
Python: Select Interpreter를 통해서 방금 우리가 생성했던 'pyqt5' 가상환경으로 변경
-> 그 후에 다시 터미널 창을 열면 가상환경에서 실행이 된다.

7. 그 후에 다음과 같은 코드의 'blank_window.py'파일을 작성한 후에

import sys

from PyQt5.QtWidgets import QApplication, QWidget

  
  

class MyApp(QWidget):

  

    def __init__(self):

        super().__init__()

        self.initUI()

  

    def initUI(self):

        self.setWindowTitle('My First Application')

        self.move(300, 300)

        self.resize(400, 200)

        self.show()

  
  

if __name__ == '__main__':

   app = QApplication(sys.argv)

   ex = MyApp()

   sys.exit(app.exec_())

8. (pyqt5) C:\Users\SBA\github\pyqt5_example>python blank_window.py
를 통해 실행을 해보자.

추가) 가상환경 삭제
conda env remove -n pyqt5

가상환경 목록 보기

conda env list