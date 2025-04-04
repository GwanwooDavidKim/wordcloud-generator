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
INPUT_FILE = 'data/grouped_keywords.json'
FONT_PATH = 'fonts/NanumGothic.ttf'
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'network_viz.png') # 원래 출력 파일 이름

# --- 시각화 조정 변수 ---
NUM_COLS = 2         # 2x2 레이아웃
MAX_NODE_SIZE = 4500 # 가장 중요한 노드 크기
MIN_NODE_SIZE = 700  # 나머지 노드 크기
MAX_NODE_ALPHA = 1.0 # 노드 투명도 (고정)
FONT_SIZE = 8
FONT_WEIGHT = 'bold'
LAYOUT_K = 0.9
LAYOUT_ITERATIONS = 150
FIG_WIDTH = 10
FIG_HEIGHT = 7

EDGE_WIDTH = 0.4
EDGE_ALPHA = 0.1

# --- 열(Column)별 기본 색상 ---
COL1_BASE_COLOR = 'dodgerblue'  # 1열(왼쪽) 기본 파랑 계열
COL2_BASE_COLOR = 'crimson'    # 2열(오른쪽) 기본 빨강 계열

# --- 색 농도 조절 계수 ---
DARK_SHADE_FACTOR = 0.6   # 중앙 노드 어둡게
LIGHT_TINT_FACTOR = 0.3   # 주변 노드 밝게 (덜 밝게 조정됨)
NODE_BORDER_COLOR = 'grey' # 노드 테두리 색상
NODE_BORDER_WIDTH = 0.5   # 노드 테두리 두께

DEFAULT_COLOR = 'grey'

# --- 유틸리티 함수 ---
def get_color_shade(base_color_rgb, factor):
    return tuple(np.clip(np.array(base_color_rgb) * factor, 0, 1))

def get_color_tint(base_color_rgb, factor):
    white_rgb = np.array([1.0, 1.0, 1.0])
    base = np.array(base_color_rgb)
    tint_rgb = base * (1 - factor) + white_rgb * factor
    return tuple(np.clip(tint_rgb, 0, 1))

def get_text_color_for_bg(bg_color_rgb):
    luminance = 0.299*bg_color_rgb[0] + 0.587*bg_color_rgb[1] + 0.114*bg_color_rgb[2]
    return 'white' if luminance < 0.5 else 'black'

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
    """하위 카테고리별 네트워크 그래프 (2x2, 열별 색상, 중요도별 명암/크기, 텍스트색상, 테두리)"""

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

    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = axes.flatten()

    print(f"서브플롯 생성: {num_rows}행 x {NUM_COLS}열")

    plot_index = 0
    for sub_category, keywords in sub_category_data.items():
        if plot_index >= len(axes):
            print("경고: 서브플롯 개수보다 하위 카테고리가 많습니다.")
            break

        ax = axes[plot_index]

        # --- <<< 열(Column) 기준으로 기본 색상 결정 >>> ---
        current_col_index = plot_index % NUM_COLS # 0 또는 1
        if current_col_index == 0: # 1열 (왼쪽)
            base_color_name = COL1_BASE_COLOR
        else: # 2열 (오른쪽)
            base_color_name = COL2_BASE_COLOR

        try:
            base_rgb = mcolors.to_rgb(base_color_name)
        except Exception:
            print(f"Warning: Invalid color name {base_color_name}. Using default.")
            base_rgb = mcolors.to_rgb(DEFAULT_COLOR)

        if not keywords:
            # ... (빈 카테고리 처리) ...
            print(f"Skipping empty sub-category: {sub_category}")
            ax.set_title(f"{sub_category} (키워드 없음)", fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
            ax.axis('off')
            plot_index += 1
            continue

        print(f"'{sub_category}' 그래프 생성 및 그리기 시작 (기본색: {base_color_name})...")

        # ... (그래프 생성, 엣지 추가, 레이아웃 계산 - 이전과 동일) ...
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

        effective_k = LAYOUT_K
        num_nodes = G_sub.number_of_nodes()
        if num_nodes > 1:
             effective_k = LAYOUT_K / (num_nodes ** 0.6)
             effective_k = max(0.05, effective_k)

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


        # 노드별 속성 계산 (크기, 색상, 텍스트 색상 - 색상 계산 시 열 기준 base_rgb 사용)
        node_attributes = {}
        num_keywords = len(keywords)

        for kw_idx, keyword in enumerate(keywords):
            importance_ratio = (kw_idx / (num_keywords - 1)) if num_keywords > 1 else 0
            importance_ratio_rev = 1.0 - importance_ratio

            current_size = MIN_NODE_SIZE
            if kw_idx == 0:
                current_size = MAX_NODE_SIZE

            # 색상 (중앙=진하게, 주변=연하게)
            if kw_idx == 0:
                current_color_rgb = get_color_shade(base_rgb, DARK_SHADE_FACTOR)
            else:
                current_color_rgb = get_color_tint(base_rgb, LIGHT_TINT_FACTOR)

            text_color = get_text_color_for_bg(current_color_rgb)

            node_attributes[keyword] = {
                'size': current_size,
                'color': current_color_rgb,
                'text_color': text_color
            }

        # NetworkX draw 함수용 리스트 준비
        node_list = list(G_sub.nodes())
        ordered_sizes = [node_attributes.get(node, {}).get('size', MIN_NODE_SIZE) for node in node_list]
        ordered_colors = [node_attributes.get(node, {}).get('color', DEFAULT_COLOR) for node in node_list]

        # 노드 및 엣지 그리기 (테두리 포함)
        nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=ordered_sizes, node_color=ordered_colors, alpha=MAX_NODE_ALPHA,
                               edgecolors=NODE_BORDER_COLOR, linewidths=NODE_BORDER_WIDTH)
        nx.draw_networkx_edges(G_sub, pos, ax=ax, width=EDGE_WIDTH, alpha=EDGE_ALPHA, edge_color='lightgrey')

        # 레이블 개별적으로 그리기
        for node, (x, y) in pos.items():
            attributes = node_attributes.get(node, {})
            text_color = attributes.get('text_color', 'black')
            current_font_size = FONT_SIZE

            ax.text(x, y, node,
                    size=current_font_size,
                    color=text_color,
                    family=font_name,
                    weight=FONT_WEIGHT,
                    ha='center',
                    va='center')

        # 서브플롯 제목 및 축 설정
        ax.set_title(sub_category, fontproperties=fm.FontProperties(fname=FONT_PATH, size=FONT_SIZE+2, weight=FONT_WEIGHT))
        ax.axis('off')
        print(f"'{sub_category}' 그래프 그리기 완료.")
        plot_index += 1

    # 남는 빈 서브플롯 숨기기
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=4.0)
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
        sub_category_data = load_grouped_keywords(INPUT_FILE)
        if sub_category_data:
            create_and_draw_subplots(sub_category_data)
        else:
            print("오류: 입력 파일에서 유효한 그룹 데이터를 읽지 못했습니다.")
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
