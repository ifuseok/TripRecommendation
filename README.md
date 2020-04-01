# TripRecommendation
- 간사이지방 여행지 추천 앱에 내장될 딥러닝 추천 모델링을 위한 코드
- 사용 알고리즘 : GRU4Rec(여행지의 attribute concat, parallel 추가 적용)- 
- 딥러넹 프레임워크 : Tensorflow 1.x version
- Tensorflow 모델링 save => Tensorflow weight를 .pb로 변환 => coreML을 이용해 .ml형태로 모델 최종 변환
- Tensorflow 를 coreML변환시 gather operator 사용 불가(GRU cell forward process low level로 직접코딩하여 해결)
