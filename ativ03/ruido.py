img1_path=r"C:\Users\LENOVO\Documents\VisaoComputacional\atividades\ativ03\img\banco2.jpg"
img2_path=r"C:\Users\LENOVO\Documents\VisaoComputacional\atividades\ativ03\img\sapato1.jpg"

""" 
================
    PARTE 1 
================  
"""
#importando as imagens e extraindo caracteristicas iniciais

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_cv2(caminho_imagem):
    img = cv2.imread(caminho_imagem, 0)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_imagem}")
    return img

def calcular_metricas_imagem(entrada):
    """Calcula os dados matemáticos e de frequência de uma imagem."""
    if isinstance(entrada, str):
        img = cv2.imread(entrada, 0)
        if img is None:
            raise FileNotFoundError(f"Não foi possível carregar: {entrada}")
    else:
        img = entrada

    media = np.mean(img)
    variancia = np.var(img)
    
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    espectro_magnitude = 20 * np.log(np.abs(f_shift) + 1)

    return {
        "imagem_original": img,
        "media": media,
        "variancia": variancia,
        "espectro_frequencia": espectro_magnitude
    }

def caracteristicas_img_plot(eixos_linha, dados_imagem, titulo_imagem):
    img = dados_imagem["imagem_original"]
    media = dados_imagem["media"]
    variancia = dados_imagem["variancia"]
    espectro = dados_imagem["espectro_frequencia"]

    eixos_linha[0].imshow(img, cmap='gray')
    eixos_linha[0].set_title(f"{titulo_imagem} - Original")
    eixos_linha[0].axis('off')
    
    texto_metricas = f"Média: {media:.2f}\nVariância: {variancia:.2f}"
    eixos_linha[0].text(0.5, -0.2, texto_metricas, transform=eixos_linha[0].transAxes,
                        ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    eixos_linha[1].hist(img.ravel(), bins=256, range=[0, 256], color='purple', alpha=0.7)
    eixos_linha[1].set_title(f"Histograma ({titulo_imagem})")
    eixos_linha[1].set_xlabel("Brilho")
    eixos_linha[1].set_ylabel("Pixels")
    eixos_linha[1].grid(True, linestyle='--', alpha=0.5)

    eixos_linha[2].imshow(espectro, cmap='gray')
    eixos_linha[2].set_title(f"FFT ({titulo_imagem})")
    eixos_linha[2].axis('off')

def exibir_plots(dados_primeira_img, nome_primeira_img, dados_segunda_img, nome_segunda_img):
    """
    Gera uma única janela contendo a matriz de subplots para as duas imagens fornecidas.
    Centraliza a criação da figura e a chamada das funções de preenchimento de linha.
    """
    fig, eixos = plt.subplots(2, 3, figsize=(14, 10))
    fig.suptitle("Métricas e Análise das Imagens", fontsize=18, fontweight='bold')

    caracteristicas_img_plot(eixos[0], dados_primeira_img, nome_primeira_img)
    caracteristicas_img_plot(eixos[1], dados_segunda_img, nome_segunda_img)

    plt.tight_layout()
    plt.show()

""" Extraindo características das imagens(tamanho, qtd. canais de cores, formato etc) """
def extrair_info(caminho_imagem):
    img = cv2.imread(caminho_imagem, 0)
    
    if img is None:
        print("Erro ao carregar imagem.")
        return

    dimensoes = img.shape
    altura = dimensoes[0]
    largura = dimensoes[1]
    
    canais = dimensoes[2] if len(dimensoes) > 2 else 1
    _, extensao = os.path.splitext(caminho_imagem)

    print(f"--- Informações ---")
    print(f"Extensão do Arquivo: {extensao.upper()}")
    print(f"Resolução: {largura}x{altura} pixels")
    print(f"Canais de Cor: {canais}")
    print("-" * 30)

def salt_and_pepper_noise(image, noise_rate):
    noisy_image = np.copy(image)
    min_value, max_value = 0, 255
    
    total_pixels = noisy_image.size
    num_noisy_pixels = int(total_pixels * noise_rate)

    noisy_pixel_indices = np.random.choice(total_pixels, num_noisy_pixels, replace=False)

    pepper_indices = noisy_pixel_indices[::2]
    noisy_image.flat[pepper_indices] = min_value

    salt_indices = noisy_pixel_indices[1::2]
    noisy_image.flat[salt_indices] = max_value

    return noisy_image

img1=parse_cv2(img1_path)
img2=parse_cv2(img2_path)

dados_banco = calcular_metricas_imagem(img1_path)
dados_sapato = calcular_metricas_imagem(img2_path)

# Testando plots
extrair_info(img1_path)
extrair_info(img2_path)
exibir_plots(dados_banco,'Banco',dados_sapato,'Sapato') #imagem original

"""
=================== 
COMENTARIOS 
===================

[Média]
> Banco: 
> Sapato: 

[Variância]
> Banco: 
> Sapato: 

[Histograma]
> Banco: 
> Sapato: 

[FFT] 
> Banco: Na imagem transformada do banco, o brilho no centro da imagem indica que uma grande parte da mesma é 
de baixa frequencia (regiões homogeneas, como a superfície branca do banco e os blocos dos azulejos).
Além disso, existem várias direçoes para essas frequências, semelhante a um asterisco.
Isso se dá pela variação na perspectiva das bordas presentes na imagem (vários ângulos diferentes).

> Sapato: 

"""

####

""" 
================
    PARTE 2 
================   
"""
#aplicando ruidos
#ruido GAUSSIANO

noise1 = np.random.normal(0, 30, img1.shape)
noise2=np.random.normal(0,30,img2.shape)

img1_ruido = np.clip(img1 + noise1, 0, 255).astype(np.uint8)
img2_ruido = np.clip(img2 + noise2, 0, 255).astype(np.uint8)

dados_banco_ruido = calcular_metricas_imagem(img1_ruido)
dados_sapato_ruido = calcular_metricas_imagem(img2_ruido)

exibir_plots(dados_banco_ruido, 'Banco com Ruído Gauss', dados_sapato_ruido, 'Sapato com Ruído Gauss') #imagem c ruido gaussiano

""" 
=================== 
COMENTARIOS 
===================

1 - parâmetros utilizados (média e desvio padrão): 
1.1 Média: 0. Indica que o ruído tem a mesma probabilidade de clarear ou escurecer um pixel, mantendo a média global de brilho da imagem praticamente inalterada
1.2 Desvio Padrao: 30. Um desvio de 30 cria uma variação perceptível, responsável por "espalhar" mais a intensidade por toda a imagem

2 - impacto visual
Houve uma degradação notável, especialmente nas bordas onde há sombra no banco. Quanto ao domínio de frequência FFT,
o "asterisco" original da imagem do banco e o "asterisco inclinado" do sapato perderam contraste, indicando que as altas frequências do 
ruído mascararam os detalhes geométricos originais da imagem.
Quanto ao histograma, No histograma original, existem picos bem definidos que representam as cores predominantes dos objetos.
A aplicação do filtro suavizou a distribuição de frequências, deixando o gráfico mais achatado devido ao espalhamento
dos pixels em torno da média.

3-O ruído está distribuído uniformemente na imagem? 
Apenas estatisticamente, conforme mostram os histogramas do segundo subplot

4-Que tipo de problema esse ruído pode causar para algoritmos de visão computacional? 
A degradação de bordas causada por esse ruído pode inviabilizar algoritmos de segmentação e de extratores de características

"""

""" 
================
    PARTE 3 
================   
"""
#aplicando ruidos
#ruido SAL E PIMENTA

img1_sp = salt_and_pepper_noise(img1, 0.2)
img2_sp = salt_and_pepper_noise(img2, 0.2)

dados_banco_sp = calcular_metricas_imagem(img1_sp)
dados_sapato_sp = calcular_metricas_imagem(img2_sp)

exibir_plots(dados_banco_sp,'Banco ruído sp',dados_sapato_sp,'Sapato ruído sp') #imagem com ruido sal e pimenta

""" 
=================== 
COMENTARIOS 
===================
ADD AQUI
"""


""" 
================
    PARTE 4 
================  
"""
#Filtro de passa baixa Gaussiano



""" 
=================== 
COMENTARIOS 
===================
->> metricas: PNSR, media, ↓ desvio padrao/variancia em ROI,Contagem Quantitativa de Pixels de Borda (Pós-Canny ou Sobel)
"""