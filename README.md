# 한국어 핵심 요약 · 원본

`app.py`는 원본 `ko_summary.ipynb`의 학습 결과를 사용하는 Streamlit 추론 화면입니다.
v2 저장소와는 별개입니다. Python 3.12, 엔트리포인트 `app.py`, 브랜치 `main`을 사용합니다.

```sh
pip install -r requirements.txt
streamlit run app.py
```

기본 관계·반복 개선, 문서 맥락 요약, 기존 문단 요약을 포함합니다. 모델은 Git LFS로 보관합니다.
입력 문서는 서버 메모리에서 처리하며 앱이 파일에 자동 저장하지 않습니다.

`ko_summary.ipynb`는 현재 로컬 노트북의 코드·마크다운을 그대로 보관하고, 기존 출력과 실행 번호만 제거했습니다.
노트북은 Streamlit 서버에서 실행하거나 재학습하지 않습니다. 재학습에는 `requirements-nlp.txt`,
별도로 확보한 AI Hub 데이터와 원본 기본 모델 및 로컬 학습 환경이 필요합니다.
학습·평가 원문, 정답 요약, 사례 데이터, MLflow 기록 및 개인 파일은 포함하지 않습니다.

출처: [gogamza/kobart-base-v2](https://huggingface.co/gogamza/kobart-base-v2),
[Huffon/klue-roberta-base-nli](https://huggingface.co/Huffon/klue-roberta-base-nli),
[AI Hub 요약문 및 레포트 생성 데이터 582](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=582).
모델별 원본 기록은 `artifacts/`의 README와 설정 파일에 있습니다.
생성 요약은 초안이며 주체·숫자·상태·핵심 내용을 원문과 대조해야 합니다.
