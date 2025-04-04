# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg') # <<< GitHub Actions 같은 Non-GUI 환경에서 필수
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import numpy as np
import matplotlib.colors as mcolors # 색상 변환 및 보간 위해 추가

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 그룹화된 키워드 JSON 파일 (중요도 순 정렬)
FONT_PATH = 'fonts/NanumGothic.ttf'      # 한글 폰트 경로 (필수!)
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png')

# --- 시각화 조정 변수 ---
NUM_COLS = 3         # 한 행에 표시할 그래프 수
MAX_NODE_SIZE = 3500 # 가장 중요한 노드 크기
MIN_NODE_SIZE = 800  # 가장 덜 중요한 노드 크기
MAX_NODE_ALPHA = 1.0 # 가장 중요한 노드 투명도 (불투명)
MIN_NODE_ALPHA = 0.4 # 가장 덜 중요한 노드 투명도 (조금 더 투명하게)
FONT_SIZE = 9        # 노드 레이블 폰트 크기
FONT_WEIGHT = 'bold' # 폰트 굵기 ('normal', 'bold')
LAYOUT_K = 0.9       # 노드 간 거리 조절
FIG_SCALE = 5        # 전체 이미지 크기 조절 (세로 크기 줄임)

EDGE_WIDTH = 0.5
EDGE_ALPHA = 0.15

# --- 행별 색상 지정 (그라데이션용) ---
ROW1_DARK_COLOR = 'royalblue'    # 1행 중요 키워드 색상 (진한 파랑 계열)
ROW1_LIGHT_COLOR = 'lightskyblue' # 1행 덜 중요한 키워드 색상 (연한 파랑 계열)
ROW2_DARK_COLOR = 'firebrick'    # 2행 중요 키워드 색상 (진한 빨강 계열)
ROW2_LIGHT_COLOR = 'lightcoral'  # 2행 덜 중요한 키워드 색상 (연한 빨강 계열)
DEFAULT_COLOR = 'grey'         # 혹시 모를 기본 색상

# --- 색상 보간 함수 ---
def linear_interpolate_color(color1_name, color2_name, fraction):
    """두 색상 이름 사이를 선형 보간하여 RGB 튜플 반환"""
    c1_rgb = np.array(mcolors.to_rgb(color1_name))
    c2_rgb = np.array(mcolors.to_rgb(color2_name))
    interpolated_rgb = c1_rgb * (1 - fraction) + c2_rgb * fraction
    # RGB 값은 0과 1 사이여야 함
    return tuple(np.clip(interpolated_rgb, 0, 1))

def load_grouped_keywords(filepath):
    """JSON 파일에서 그룹화된 키워드 로드 (중요도 순으로 정렬된 리스트 기대)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"그룹화된 키워드 파일 로드 완료: {filepath}")
        sub_category_count = sum(len(sub_cats) for sub_cats in data.values())
        print(f"총 하위 카테고리 개수: {sub_category_count}")
        if sub_category_count != NUM_COLS * 2:
             print(f"경고: 하위 카테고리 개수({sub_category_count})가 6개가 아닙니다. 레이아웃이 예상과 다를 수 있습니다.")
        return data, sub_category_count
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise
    except json.JSONDecodeError:
        print(f"오류: '{filepath}' 파일이 유효한 JSON 형식이 아닙니다.")
        raise
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise

def create_and_draw_subplots(grouped_data, num_sub_categories):
    """하위 카테고리별 네트워크 그래프를 서브플롯에 그리기 (중요도, 색상 그라데이션 반영)"""

    # --- 한글 폰트 설정 ---
    try:
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
        font_prop = fm.FontProperties(fname=FONT_PATH)
        font_name = font_prop.get_name()
        fm.fontManager.addfont(FONT_PATH)
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)
        print(f"폰트 설정 완료: {FONT_PATH} (Family: {font_name})")
    except Exception as e:
        print(f"오류: 폰트 파일을 설정할 수 없습니다. '{FONT_PATH}'. 오류: {e}")
        raise FileNotFoundError(f"Font file not found or invalid: {FONT_PATH}")
    # --- 폰트 설정 끝 ---

    num_rows = math.ceil(num_sub_categories / NUM_COLS)
    if num_rows != 2 or num_sub_categories > 6:
        print("경고: 2행 3열(총 6개) 레이아웃을 벗어납니다. 색상 및 배치가 이상할 수 있습니다.")
        num_rows = max(2, num_rows)

    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(NUM_COLS * FIG_SCALE, num_rows * FIG_SCALE))
    axes = axes.flatten()

    print(f"서브플롯 생성: {num_rows}행 x {NUM_COLS}열")

    plot_index = 0
    for main_category, sub_categories in grouped_data.items():
        for sub_category, keywords in sub_categories.items():
            if plot_index >= len(axes):
                print("경고: 서브플롯 개수보다 하위 카테고리가 많습니다.")
                break

            ax = axes[plot_index]

            # 행에 따른 기본 색상 (그라데이션 시작/끝 색상) 결정
            if plot_index < NUM_COLS: # 1행
                dark_color = ROW1_DARK_COLOR
                light_color = ROW1_LIGHT_COLOR
            elif plot_index < NUM_COLS * 2: # 2행
                dark_color = ROW2_DARK_COLOR
                light_color = ROW2_LIGHT_COLOR
            else:
                dark_color = DEFAULT_COLOR
                light_color = DEFAULT_COLOR

            if not keywords:
                print(f"Skipping empty sub-category: {sub_category}")
                ax.set_title(f"{sub_category} (키워드 없음)", fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
                ax.axis('off')
                plot_index += 1
                continue

            print(f"'{sub_category}' 그래프 생성 및 그리기 시작...")

            G_sub = nx.Graph()
            for keyword in keywords:
                 G_sub.add_node(keyword)

            keyword_list_for_edges = list(keywords)
            for i in range(len(keyword_list_for_edges)):
                for j in range(i + 1, len(keyword_list_for_edges)):
                    G_sub.add_edge(keyword_list_for_edges[i], keyword_list_for_edges[j])

            effective_k = LAYOUT_K
            if G_sub.number_of_nodes() > 1:
                 effective_k = LAYOUT_K / np.log(G_sub.number_of_nodes() + 1)
                 effective_k = max(0.1, effective_k)

            pos = nx.spring_layout(G_sub, k=effective_k, iterations=100, seed=42)

            # --- 노드별 크기, 알파, 색상 계산 ---
            keyword_to_size = {}
            keyword_to_alpha = {}
            keyword_to_color = {}
            num_keywords = len(keywords)

            for kw_idx, keyword in enumerate(keywords):
                # 중요도 비율 계산 (0: 가장 중요, 1: 가장 덜 중요)
                importance_ratio = (kw_idx / (num_keywords - 1)) if num_keywords > 1 else 0

                # 크기 계산 (선형 감소)
                current_size = MAX_NODE_SIZE - (MAX_NODE_SIZE - MIN_NODE_SIZE) * importance_ratio
                keyword_to_size[keyword] = current_size

                # 알파 계산 (선형 감소)
                current_alpha = MAX_NODE_ALPHA - (MAX_NODE_ALPHA - MIN_NODE_ALPHA) * importance_ratio
                keyword_to_alpha[keyword] = current_alpha

                # 색상 계산 (선형 보간)
                current_color_rgb = linear_interpolate_color(dark_color, light_color, importance_ratio)
                keyword_to_color[keyword] = current_color_rgb

            # NetworkX draw 함수용 리스트 준비 (G_sub.nodes() 순서 기준)
            node_list = list(G_sub.nodes())
            ordered_sizes = [keyword_to_size.get(node, MIN_NODE_SIZE) for node in node_list]
            ordered_alphas = [keyword_to_alpha.get(node, MIN_NODE_ALPHA) for node in node_list]
            ordered_colors = [keyword_to_color.get(node, mcolors.to_rgb(DEFAULT_COLOR)) for node in node_list]
            # --- 계산 끝 ---

            # 그리기 (계산된 리스트 사용)
            nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=ordered_colors, alpha=MAX_NODE_ALPHA) # 기본 알파는 최대로 하고, 개별 노드는 아래 레이블에서 조절하거나 노드 자체 알파 사용 가능
            # nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=ordered_colors, alpha=ordered_alphas) # 노드별 알파 직접 적용 시

            nx.draw_networkx_edges(G_sub, pos, ax=ax, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')

            # 레이블 그리기 (굵게)
            nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=FONT_SIZE, font_family=font_name, font_weight=FONT_WEIGHT) # font_weight 추가

            # 서브플롯 제목 설정 (굵게) 및 축 숨기기
            ax.set_title(sub_category, fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT)) # 제목도 bold 적용
            ax.axis('off')
            print(f"'{sub_category}' 그래프 그리기 완료.")
            plot_index += 1

    # 남는 빈 서브플롯 숨기기
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=3.0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"전체 네트워크 시각화 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
    except Exception as e:
        print(f"ERROR saving image file: {e}")
        raise e
    finally:
        plt.close(fig)

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        grouped_data, num_sub_categories = load_grouped_keywords(INPUT_FILE)
        if grouped_data and num_sub_categories > 0:
            create_and_draw_subplots(grouped_data, num_sub_categories)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했거나 하위 카테고리가 없습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
