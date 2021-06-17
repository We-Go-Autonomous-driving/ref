# 🏎AIFFEL 대전 1기 자율주행 프로젝트🏎
자율주행, 협동로봇 플랫폼을 제공하는 기업 위고 코리아와 협업한 프로젝트 [위고코리아 홈페이지](https://wego-robotics.com/)

## 1. 팀명: We-Go
## 2. 일정: 2021.05.10 ~ 2021.06.18 (약 6주)
## 3. 팀원: 양창원(팀장), 김강태, 임진선, 안석현, 문재윤
## 4. 목표: 실내 자율주행 모바일 로봇이 특정 인물을 Tracking 할 수 있는 기능 구현
## 5. 역할
|역할|main|sub|
|---|---|---|
|Tracking(Siam)|김강태|-|
|Tracking(DeepSORT)|양창원|-|
|depth camera|안석현|김강태|
|ROS|문재윤|양창원|
|H/W|문재윤|임진선|
|Recording|임진선|-|

### [프로젝트 진행 Notion](https://www.notion.so/We-Go-ed512708c2f14177a53e4f5c95d918a9)

# Code 사용 방법
## 설치(Installation)
1. ROS
2. scout-mini
3. yolov4-deepsort

## 1. ROS 설치 & workspace init
1) [ROS 설치 링크](http://wiki.ros.org/melodic/Installation/Ubuntu)로 이동  
2) ROS Melodic(Ubuntu 18.04 호환 버전) 설치  
3) update까지 마치고 desktop-full 실행  
`$ sudo apt install ros-melodic-desktop-full`  
4) 1.6.1까지 진행  
5) 설치 후 터미널에서 `roscore` 실행으로 정상적으로 설치되었는지 확인  

![roscore](image/roscore.png)

참고) ROS Melodic에서 Python3를 사용하기 위해서는 아래 명령어 입력 필요  
`$ sudo apt-get install python3-catkin-pkg-modules`  
`$ sudo apt-get install python3-rospkg-modules`

6) 터미널 창에서 아래와 같이 작업 공간(폴더)를 생성한다. (catkin_ws 이외에 다른 폴더 이름을 해도 상관없다.)
`$ cd ~ && mkdir -p catkin_ws/src`
`$ cd ~/catkin_ws/src`
7) workspace init 실시
`$ catkin_init_workspace`

## 2. scout-mini 설치
1) 위 내용과 이어짐. scout-mini github code를 clone해야 한다. ROS workspace init한 상태에서 바로 진행한다.
`$ git clone https://github.com/agilexrobotics/scout_mini_ros.git`
2) 새로운 패키지(폴더)를 설치하면 catkin_make를 해줘야 한다.(상위 폴더에서 해야함)  
`$ cd .. && catkin_make`

참고) Python 파일을 새로 생성한 후에는 해당 파일의 권한 설정이 필요하다.  
`$ sudo chmod +x (파일이름)`
또는 모든 파일에 대해서 한 번에 할 때는 아래와 같은 명령어 사용  
`$ sudo chmod +x ./*`  

여기까지 하면 scout-mini를 제어할 수 있는 단계가 된다.  

## 3. yolov4-deepsort 설치
추적 알고리즘인 deepsort 중에서 yolo
