
import os
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
# Import necessário para as características de textura
from skimage.feature import graycomatrix, graycoprops

def extract_robust_features(image_path):
    """
    Extrai um vetor de características combinando ABCDE (com segmentação melhorada) e Textura.
    """
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # --- SEGMENTAÇÃO COM LIMIAR ADAPTATIVO ---
        # Em vez de um limiar global (Otsu), o limiar adaptativo se ajusta a diferentes condições de iluminação.
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        main_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

        # --- EXTRAÇÃO DE CARACTERÍSTICAS "ABCDE" ---
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        if perimeter == 0: return None
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if len(main_contour) < 5:
            major_axis = 0
        else:
            _, (major_axis, _), _ = cv2.fitEllipse(main_contour)

        # --- ADIÇÃO DE CARACTERÍSTICAS DE TEXTURA (GLCM) ---
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        texture_features = [contrast, dissimilarity, homogeneity, energy, correlation]

        # --- CARACTERÍSTICAS DE COR ---
        (R, G, B) = cv2.split(image_rgb)
        color_features = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]

        # Concatena o novo conjunto robusto de características
        all_features = np.concatenate([
            [area, circularity, major_axis],
            texture_features,
            color_features
        ])

        return all_features

    except Exception as e:
        # print(f"Erro ao processar {os.path.basename(image_path)}: {e}")
        return None

def process_dataset(image_folder, metadata_csv, output_path):
    """
    Processa um conjunto de dados, extrai características avançadas e salva o resultado.
    """
    print(f"\nProcessando dataset de '{image_folder}'...")
    metadata = pd.read_csv(metadata_csv)
    feature_list, label_list = [], []

    for _, row in metadata.iterrows():
        img_id, label = row['image'], row['MEL']
        image_path = os.path.join(image_folder, f"{img_id}.jpg")

        if os.path.exists(image_path):
            features = extract_robust_features(image_path)
            if features is not None:
                feature_list.append(features)
                label_list.append(label)

    feature_df = pd.DataFrame(feature_list)
    feature_df['label'] = label_list
    feature_df.to_csv(output_path, index=False)
    print(f"Arquivo de características salvo em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai um conjunto robusto de características de imagens médicas.")
    parser.add_argument('--input_dir', required=True, help='Diretório base contendo os dados originais.')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar os arquivos CSV com as características.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_dataset(
        image_folder=os.path.join(args.input_dir, 'treino_images'),
        metadata_csv=os.path.join(args.input_dir, 'treino.csv'),
        output_path=os.path.join(args.output_dir, 'train_features.csv')
    )

    process_dataset(
        image_folder=os.path.join(args.input_dir, 'validacao_images'),
        metadata_csv=os.path.join(args.input_dir, 'validacao.csv'),
        output_path=os.path.join(args.output_dir, 'validation_features.csv')
    )

    process_dataset(
        image_folder=os.path.join(args.input_dir, 'teste_images'),
        metadata_csv=os.path.join(args.input_dir, 'teste.csv'),
        output_path=os.path.join(args.output_dir, 'test_features.csv')
    )

    print("\nExtração de características robustas concluída.")
