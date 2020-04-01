## 여행경로 추천 모델
- 사용자가 처음에 선택한 여행지를 input으로 받아 다음 여행지를 추천하는 모델
- 학습된 모델을 앱에 내장될 수 있도록 변환 하는 작업포함
### 사용 알고리즘
- GRU4Rec(여행지의 attribute concat, parallel 추가 적용)
- 참조 논문 제목:  SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS
- https://arxiv.org/pdf/1511.06939.pdf
### 딥러넹 프레임워크
- Tensorflow 1.x version
### 모델링 과정
1. Tensorflow 모델링 and save 
2. Tensorflow weight를 .pb로 변환
3. coreML을 이용해 .ml형태로 모델 최종 변환
### 제약 사항 및 해결
- Tensorflow 를 coreML변환시 gather operator 사용 불가 => GRU cell forward process low level로 직접코딩하여 해결
