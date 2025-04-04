# generate_network_viz.py
import json
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import numpy as np
import matplotlib.colors as mcolors

# --- 설정값 ---
INPUT_FILE = 'data/grouped_keywords.json' # 기존 JSON 파일 이름
FONT_PATH = 'fonts/NanumGothic.ttf'
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png') # 기존 출력 파일 이름

# --- 시각화 조정 변수 ---
NUM_COLS = 2         # 2x2 레이아웃
MAX_NODE_SIZE = 3200 # 최대 노드 크기 약간 감소
MIN_NODE_SIZE = 900
MAX_NODE_ALPHA = 1.0
MIN_NODE_ALPHA = 0.4
FONT_SIZE = 8
FONT_WEIGHT = 'bold'
LAYOUT_K = 0.9       # 노드 간 기본 거리 약간 감소
LAYOUT_ITERATIONS = 150
# FIG_SCALE 대신 figsize 직접 설정
FIG_WIDTH = 10       # 전체 이미지 가로 크기
FIG_HEIGHT = 7       # 전체 이미지 세로 크기 (가로보다 작게 설정)

EDGE_WIDTH = 0.5
EDGE_ALPHA = 0.15

# --- 사용할 컬러맵 ---
COLOR_MAP_NAME = 'plasma' # <<< plasma 컬러맵 사용
DEFAULT_COLOR = 'grey'

# --- 색상 보간 불필요 (컬러맵 직접 사용) ---

def load_grouped_keywords(filepath):
    # ... (이전과 동일) ...
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"그룹화된 키워드 파일 로드 완료: {filepath}")
        if len(data) == 1:
            main_key = list(data.keys())[0]
            sub_category_data = data[main_key]
            sub_category_count = len(sub_category_data)
            print(f"총 하위 카테고리 개수: {sub_category_count} (in '{main_key}')")
            if sub_category_count != NUM_COLS * 2:
                 print(f"경고: 하위 카테고리 개수({sub_category_count})가 4개가 아닙니다. 레이아웃/색상 오류 가능성.")
            return sub_category_data
        else:
            print("오류: JSON 파일에 예상된 단일 최상위 카테고리가 없습니다.")
            raise ValueError("JSON format error: Expected a single top-level key.")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{filepath}'을 찾을 수 없습니다.")
        raise
    except json.JSONDecodeError:
        print(f"오류: '{filepath}' 파일이 유효한 JSON 형식이 아닙니다.")
        raise
    except Exception as e:
        print(f"파일 읽기/파싱 중 오류 발생: {e}")
        raise

def create_and_draw_subplots(sub_category_data):
    """하위 카테고리별 네트워크 그래프 그리기 (2x2, plasma 컬러맵, 중요도 반영)"""

    num_sub_categories = len(sub_category_data)
    num_rows = math.ceil(num_sub_categories / NUM_COLS)

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

    # <<< figsize 직접 지정하여 직사각형 형태 만들기 >>>
    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = axes.flatten()

    print(f"서브플롯 생성: {num_rows}행 x {NUM_COLS}열")

    # --- 컬러맵 가져오기 ---
    cmap = plt.get_cmap(COLOR_MAP_NAME)

    plot_index = 0
    for sub_category, keywords in sub_category_data.items():
        if plot_index >= len(axes):
            print("경고: 서브플롯 개수보다 하위 카테고리가 많습니다.")
            break

        ax = axes[plot_index]

        if not keywords:
            # ... (빈 카테고리 처리) ...
            print(f"Skipping empty sub-category: {sub_category}")
            ax.set_title(f"{sub_category} (키워드 없음)", fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
            ax.axis('off')
            plot_index += 1
            continue

        print(f"'{sub_category}' 그래프 생성 및 그리기 시작 (컬러맵: {COLOR_MAP_NAME})...")

        G_sub = nx.Graph()
        center_node = None
        if keywords:
            center_node = keywords[0]
            for keyword in keywords:
                G_sub.add_node(keyword)

        keyword_list_for_edges = list(keywords)
        for i in range(len(keyword_list_for_edges)):
            for j in range(i + 1, len(keyword_list_for_edges)):
                G_sub.add_edge(keyword_list_for_edges[i], keyword_list_for_edges[j])

        # 레이아웃 계산 (중앙 노드 고정)
        effective_k = LAYOUT_K
        if G_sub.number_of_nodes() > 1:
             effective_k = LAYOUT_K / np.log(G_sub.number_of_nodes() + 1)
             effective_k = max(0.1, effective_k)

        fixed_positions = {}
        initial_pos = {}
        fixed_nodes = []
        if center_node and center_node in G_sub:
            fixed_positions[center_node] = (0, 0)
            initial_pos[center_node] = (0, 0)
            fixed_nodes = [center_node]

        pos = nx.spring_layout(G_sub, k=effective_k,
                               pos=initial_pos if initial_pos else None,
                               fixed=fixed_nodes if fixed_nodes else None,
                               iterations=LAYOUT_ITERATIONS,
                               seed=42)

        # 노드별 크기, 알파, 색상 계산
        keyword_to_size = {}
        keyword_to_alpha = {}
        keyword_to_color = {}
        num_keywords = len(keywords)

        for kw_idx, keyword in enumerate(keywords):
            # 중요도 비율 (0: 가장 중요, 1: 가장 덜 중요)
            # 컬러맵 적용 시 비율을 반대로 해야할 수 있음 (plasma는 0이 어둡고 1이 밝음)
            # 중요할수록 0에 가깝게 (어두운 색), 덜 중요할수록 1에 가깝게 (밝은 색)
            importance_ratio_for_color = (kw_idx / (num_keywords - 1)) if num_keywords > 1 else 0
            # 크기와 알파는 중요할수록 크게/불투명하게 (비율 반대로 사용)
            importance_ratio_for_size_alpha = 1.0 - importance_ratio_for_color

            current_size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * importance_ratio_for_size_alpha
            keyword_to_size[keyword] = current_size

            current_alpha = MIN_NODE_ALPHA + (MAX_NODE_ALPHA - MIN_NODE_ALPHA) * importance_ratio_for_size_alpha
            keyword_to_alpha[keyword] = current_alpha

            # <<< plasma 컬러맵에서 색상 가져오기 >>>
            current_color_rgba = cmap(importance_ratio_for_color)
            keyword_to_color[keyword] = current_color_rgba

        # NetworkX draw 함수용 리스트 준비
        node_list = list(G_sub.nodes())
        ordered_sizes = [keyword_to_size.get(node, MIN_NODE_SIZE) for node in node_list]
        # ordered_alphas = [keyword_to_alpha.get(node, MIN_NODE_ALPHA) for node in node_list] # 알파 개별 적용 시
        ordered_colors = [keyword_to_color.get(node, mcolors.to_rgb(DEFAULT_COLOR)) for node in node_list]

        # 그리기
        # 노드 알파는 MAX_NODE_ALPHA로 고정, 색/크기로 중요도 표현
        nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=ordered_colors, alpha=MAX_NODE_ALPHA)
        nx.draw_networkx_edges(G_sub, pos, ax=ax, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='grey')
        nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=FONT_SIZE, font_family=font_name, font_weight=FONT_WEIGHT)

        ax.set_title(sub_category, fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
        ax.axis('off')
        print(f"'{sub_category}' 그래프 그리기 완료.")
        plot_index += 1

    # 남는 빈 서브플롯 숨기기
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    # <<< tight_layout 패딩 증가시켜 잘림 방지 시도 >>>
    plt.tight_layout(pad=4.0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"전체 네트워크 시각화 이미지가 '{OUTPUT_FILE}'로 저장되었습니다.")
    except Exception as e:
        print(f"ERROR saving image file: {e}")
        raise e
    finally:
        plt.close(fig) # 메모리 해제

def main():
    """메인 실행 함수"""
    print(f"키워드 입력 파일: {INPUT_FILE}")
    try:
        sub_category_data = load_grouped_keywords(INPUT_FILE)
        if sub_category_data:
            create_and_draw_subplots(sub_category_data)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
