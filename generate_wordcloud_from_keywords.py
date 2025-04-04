# generate_wordcloud_from_keywords.py
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # 마스크 사용 시 필요

# --- 설정값 ---
INPUT_FILE = 'data/input_keywords.txt' # 키워드와 빈도수(중요도)가 저장된 파일
FONT_PATH = 'fonts/NanumGothic.ttf'   # 레포지토리 내 폰트 파일 경로 (나눔고딕 추천)
MASK_FILE = 'masks/cloud_mask.png'    # <<< 마스크 이미지 파일 경로 (추가됨)
OUTPUT_DIR = 'output'                 # 출력 이미지를 저장할 디렉토리
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'wordcloud.png') # <<< 출력 파일 이름 변경 (선택사항)
BG_COLOR = 'white'                    # 배경색
WIDTH = 800                           # 이미지 너비 (마스크 사용 시 무시될 수 있음)
HEIGHT = 600                          # 이미지 높이 (마스크 사용 시 무시될 수 있음)
COLORMAP = 'Reds'                    # <<< 단어 색상 스타일 ('Blues'로 변경됨)

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

def load_mask_image(filepath):
    """마스크 이미지 로드"""
    mask = None
    if filepath and os.path.exists(filepath):
        try:
            mask = np.array(Image.open(filepath))
            # 마스크 이미지가 너무 작거나 크면 경고 (선택적)
            if mask.shape[0] < 10 or mask.shape[1] < 10:
                print(f"경고: 마스크 이미지 '{filepath}'가 너무 작습니다 ({mask.shape}). 결과가 이상할 수 있습니다.")
            print(f"마스크 이미지 '{filepath}' 로드 성공.")
        except FileNotFoundError:
            print(f"경고: 마스크 이미지 '{filepath}'를 찾을 수 없어 마스크 없이 진행합니다.")
            mask = None
        except Exception as e:
            print(f"마스크 이미지 로드 중 오류 발생: {e}")
            mask = None
    else:
        print("마스크 파일 경로가 지정되지 않았거나 파일을 찾을 수 없습니다. 마스크 없이 진행합니다.")
    return mask

def generate_and_save_wordcloud(frequencies):
    """워드 클라우드 생성 및 저장"""
    # 폰트 파일 존재 확인
    if not os.path.exists(FONT_PATH):
        print(f"오류: 폰트 파일 '{FONT_PATH}'를 찾을 수 없습니다.")
        raise FileNotFoundError(f"Font file not found: {FONT_PATH}")

    if not frequencies:
        print("경고: 입력된 키워드가 없습니다. 빈 이미지가 생성될 수 있습니다.")
        frequencies['키워드없음'] = 1 # 예시

    # 마스크 이미지 로드 <<< 분리된 함수 호출
    mask = load_mask_image(MASK_FILE)

    print("워드 클라우드 생성 시작...")
    try:
        wc = WordCloud(font_path=FONT_PATH,
                       width=WIDTH if mask is None else None,   # <<< 마스크 있으면 너비/높이 None 설정 가능 (자동)
                       height=HEIGHT if mask is None else None, # <<< 마스크 있으면 너비/높이 None 설정 가능 (자동)
                       background_color=BG_COLOR,
                       colormap=COLORMAP,       # <<< 설정된 Colormap 사용
                       mask=mask,               # <<< 로드된 마스크 전달
                       contour_width=1 if mask is not None else 0, # 외곽선 두께 (필요시 조절)
                       contour_color='steelblue' if mask is not None else None, # 외곽선 색상 (필요시 조절)
                       # 추가 옵션 (필요에 따라 조절)
                       max_words=200,          # 최대 단어 수
                       prefer_horizontal=0.9,  # 가로 단어 비율
                       scale=1.5               # 단어 크기 스케일
                      )

        # 미리 계산된 빈도수(중요도)로 워드 클라우드 생성
        wordcloud_image = wc.generate_from_frequencies(frequencies)
        print("워드 클라우드 생성 완료.")

        # 출력 디렉토리 생성 (없으면)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 이미지 저장
        plt.figure(figsize=(10, 8)) # 그림 크기 조절 (인치 단위)
        plt.imshow(wordcloud_image, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(OUTPUT_FILE)
        print(f"워드 클라우드 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
        plt.close() # 메모리 해제를 위해 창 닫기

    except Exception as e:
        print(f"워드 클라우드 생성/저장 중 오류 발생: {e}")
        raise e

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    print(f"마스크 파일: {MASK_FILE if MASK_FILE else '미사용'}")
    print(f"폰트 파일: {FONT_PATH}")
    print(f"출력 파일: {OUTPUT_FILE}")
    print(f"색상 테마: {COLORMAP}")

    # 필요한 라이브러리 설치 확인 (선택적이지만 사용자에게 유용)
    try:
        import wordcloud
        import matplotlib
        import numpy
        import PIL
    except ImportError as e:
        print(f"\n오류: 필요한 라이브러리가 설치되지 않았습니다: {e}")
        print("pip install wordcloud matplotlib numpy Pillow")
        return # 실행 중단

    try:
        frequencies = parse_keyword_file(INPUT_FILE)
        if frequencies:
             # 읽어온 키워드/빈도수 상위 일부 출력 (로그 확인용)
            print("\n--- 입력된 키워드/빈도수 (상위 30개) ---")
            sorted_freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
            for word, freq in sorted_freq[:30]:
                print(f"{word}: {freq}")
            print("------------------------------------\n")

            generate_and_save_wordcloud(frequencies)
        else:
            print("오류: 입력 파일에서 유효한 키워드/빈도수를 읽지 못했습니다.")

    except FileNotFoundError as e:
         # 파일 경로 오류는 main에서 처리하여 구체적 메시지 제공
         print(f"\n파일 경로 오류: {e}")
         print("INPUT_FILE, FONT_PATH, MASK_FILE 경로가 올바른지 확인하세요.")
    except Exception as e:
        print(f"\n스크립트 실행 중 예상치 못한 오류 발생: {e}")

if __name__ == "__main__":
    main()
