# generate_wordcloud_from_keywords.py
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # 마스크 사용 시 필요

# --- 설정값 ---
INPUT_FILE = 'data/input_keywords.txt' # 키워드와 빈도수(중요도)가 저장된 파일
FONT_PATH = 'fonts/NanumGothic.ttf'   # 레포지토리 내 폰트 파일 경로 (나눔고딕 추천)
OUTPUT_DIR = 'output'                 # 출력 이미지를 저장할 디렉토리
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'wordcloud.png') # 최종 저장될 이미지 파일 경로
BG_COLOR = 'white'                    # 배경색
WIDTH = 800                           # 이미지 너비
HEIGHT = 600                          # 이미지 높이
COLORMAP = 'viridis'                  # 단어 색상 스타일

def parse_keyword_file(filepath):
    """키워드 파일 파싱하여 딕셔너리로 반환"""
    frequencies = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    keyword = parts[0]
                    try:
                        frequency = int(parts[1])
                        frequencies[keyword] = frequency
                    except ValueError:
                        print(f"경고: '{line}' - 빈도수 변환 오류, 건너<0xEB><0>니다.")
                else:
                    print(f"경고: '{line}' - 형식이 잘못되었습니다(키워드 빈도수), 건너<0xEB><0>니다.")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise # 오류를 다시 발생시켜 워크플로우 실패 유도
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise
    return frequencies

def generate_and_save_wordcloud(frequencies):
    """워드 클라우드 생성 및 저장"""
    # 폰트 파일 존재 확인
    if not os.path.exists(FONT_PATH):
        print(f"오류: 폰트 파일 '{FONT_PATH}'를 찾을 수 없습니다.")
        raise FileNotFoundError(f"Font file not found: {FONT_PATH}")

    if not frequencies:
        print("경고: 입력된 키워드가 없습니다. 빈 이미지가 생성될 수 있습니다.")
        # 빈 결과 처리 (예: 빈 이미지 생성 방지 또는 기본 메시지 추가)
        frequencies['키워드없음'] = 1 # 예시

    print("워드 클라우드 생성 시작...")
    try:
        # # 선택사항: 마스크 이미지 사용
        # mask_image_path = 'path/to/your/mask.png'
        # try:
        #     mask = np.array(Image.open(mask_image_path))
        # except FileNotFoundError:
        #     print(f"경고: 마스크 이미지 '{mask_image_path}'를 찾을 수 없어 마스크 없이 진행합니다.")
        #     mask = None
        mask = None # 마스크 미사용

        wc = WordCloud(font_path=FONT_PATH,
                       width=WIDTH,
                       height=HEIGHT,
                       background_color=BG_COLOR,
                       colormap=COLORMAP,
                       mask=mask,
                       contour_width=1 if mask is not None else 0,
                       contour_color='steelblue' if mask is not None else None
                      )

        # 미리 계산된 빈도수(중요도)로 워드 클라우드 생성
        wordcloud_image = wc.generate_from_frequencies(frequencies)
        print("워드 클라우드 생성 완료.")

        # 출력 디렉토리 생성 (없으면)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 이미지 저장
        plt.figure(figsize=(WIDTH/100, HEIGHT/100))
        plt.imshow(wordcloud_image, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(OUTPUT_FILE)
        print(f"워드 클라우드 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
        plt.close()

    except Exception as e:
        print(f"워드 클라우드 생성/저장 중 오류 발생: {e}")
        raise e

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        frequencies = parse_keyword_file(INPUT_FILE)
        if frequencies:
             # 읽어온 키워드/빈도수 상위 일부 출력 (로그 확인용)
            print("\n--- 입력된 키워드/빈도수 (상위 30개) ---")
            sorted_freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
            for word, freq in sorted_freq[:30]:
                print(f"{word}: {freq}")
            generate_and_save_wordcloud(frequencies)
        else:
            print("오류: 입력 파일에서 유효한 키워드/빈도수를 읽지 못했습니다.")

    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
