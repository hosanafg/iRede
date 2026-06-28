# IMPORTANDO BIBLIOTECAS
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import morphology

# DEFININDO FUNÇÕES
def extrair_info(img):   
    if img is None:
        print("Erro ao carregar imagem.")
        return

    dimensoes = img.shape
    altura = dimensoes[0]
    largura = dimensoes[1]
    
    canais = dimensoes[2] if len(dimensoes) > 2 else 1

    print(f"--- Informações ---")
    print(f"Resolução: {largura}x{altura} pixels")
    print(f"Canais de Cor: {canais}")
    print("-" * 30)

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
    fig, eixos = plt.subplots(2, 3, figsize=(12,8))
    fig.suptitle("Métricas e Análise das Imagens", fontsize=18, fontweight='bold')

    caracteristicas_img_plot(eixos[0], dados_primeira_img, nome_primeira_img)
    caracteristicas_img_plot(eixos[1], dados_segunda_img, nome_segunda_img)

    plt.tight_layout()
    plt.show()

def suavizacao_gaussiana_grande(img_ruidosa):
    """Aplica filtro Gaussiano com kernel maior 7x7 (Suavização Intensa)."""
    img_filtrada = cv2.GaussianBlur(img_ruidosa, (7, 7), 0)
    return {"imagem_filtrada": img_filtrada}

def suavizacao_gaussiana_pequena(img_ruidosa):
    """Aplica filtro Gaussiano com kernel pequeno 3x3 (Suavização Leve)."""
    img_filtrada = cv2.GaussianBlur(img_ruidosa, (3, 3), 0)
    return {"imagem_filtrada": img_filtrada}

def filtrar_mediana(img_cinza, ksize=7):
    return cv2.medianBlur(img_cinza, ksize)

def morfologia_abertura(img_binaria, tamanho_kernel=3):
    """
    Aplica a operação de Abertura (Erosão seguida de Dilação).
    Excelente para eliminar ruídos brancos isolados no fundo (limpar o fundo).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho_kernel, tamanho_kernel))
    img_processada = cv2.morphologyEx(img_binaria, cv2.MORPH_OPEN, kernel)
    return img_processada

def morfologia_fechamento(img_binaria, tamanho_kernel=3):
    """
    Aplica a operação de Fechamento (Dilação seguida de Erosão).
    Excelente para preencher buracos pretos internos e conectar fendas no objeto.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanho_kernel, tamanho_kernel))
    img_processada = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, kernel)
    return img_processada

def extrair_area(prop_regiao):
    """Função 1: Extrai a área (número de pixels) do objeto."""
    return prop_regiao.area

def extrair_perimetro(prop_regiao):
    """Função 2: Extrai o perímetro do contorno do objeto."""
    return prop_regiao.perimeter

def extrair_centroide(prop_regiao):
    """Função 3: Extrai o centro de massa (Y, X) do objeto."""
    y, x = prop_regiao.centroid
    return f"({x:.1f}, {y:.1f})"

def extrair_momentos_hu(prop_regiao):
    """
    Função 4: Extrai os Momentos de Hu (invariantes a escala, rotação e translação).
    Retorna o primeiro momento (hu[0]) como métrica de forma principal.
    """
    hu_momentos = prop_regiao.moments_hu
    return hu_momentos[0]

def export_to_csv(nome):
    try:
        df_caracteristicas.to_csv(nome_arquivo, index=False)
        
        print(f"\n[DEBUG] [SUCESSO] Arquivo gerado com sucesso!")
        print(f"[INFO] Localização: {os.path.abspath(nome_arquivo)}")

    except PermissionError:
        print(f"\n[DEBUG] [ERRO] Não foi possível salvar '{nome_arquivo}'.")

    except Exception as e:
        print(f"\n[DEBUG] [FALHA] Falha inesperada ao tentar criar o arquivo CSV.")
        print(f"[DEBUG] Detalhes do erro: {e}")


""" 
====================
PARTE 1
====================
"""
# SOBRE A IMG ESCOLHDA
imagem= data.camera()
imagem_info=extrair_info(imagem)

#Exibindo métricas da imagem
#exibir_plots(imagem_metricas,"Cameraman")

""" 
====================
PARTE 2
====================
"""

#pre-processamento: suavizacao gauss 3x3
#imagem_pre_process=filtrar_mediana(imagem)
imagem_pre_process=suavizacao_gaussiana_pequena(imagem)
#imagem_pre_process=suavizacao_gaussiana_grande(imagem)

# cálculo do limiar
#limiar = threshold_otsu(imagem)
limiar=threshold_otsu(imagem_pre_process["imagem_filtrada"])

# segmentação binária
#segmentada = imagem > limiar
#segmentada = (imagem_pre_process["imagem_filtrada"] > limiar).astype(np.uint8) * 255 --> sem inv. binária
segmentada = (imagem_pre_process["imagem_filtrada"] <= limiar).astype(np.uint8) * 255 #com inversão binária

#plots
dados_imagem=calcular_metricas_imagem(imagem)
dados_segmentada=calcular_metricas_imagem(segmentada)
exibir_plots(dados_imagem,'Imagem Original',dados_segmentada,'Imagem Limearizada (OTSU)')

""" 
==============
COMENTÁRIOS
==============

* explicar por que o método foi escolhido *
A limearização global foi escolhida por ser a mais vantajosa em termos de custo computacional, tendo em vista que 
o fundo e a imagem do cameraman possuem alto contraste entre si - o que facilita a busca pelo limiar ideal de OTSU.
Sem a suavização gaussiana (kernel 3x3), mais elementos do fundo estavam sendo capturados pelo threshold. Após o pré-
-processamento, apenas as sombras dos prédios estão sendo detectadas, o que representa uma melhora considerável em favor
da etapa de pré-processamento. No entanto, infelizmente, o contorno da câmera acabou se perdendo.

"""

""" 
====================
PARTE 3
====================
"""
#operações MORFOLÓGICAS
imagem_abertura = morfologia_abertura(segmentada, tamanho_kernel=3)
imagem_fechamento = morfologia_fechamento(imagem_abertura, tamanho_kernel=3)

#dados_original = calcular_metricas_imagem(imagem)
dados_abertura=calcular_metricas_imagem(imagem_abertura)
dados_morfologia = calcular_metricas_imagem(imagem_fechamento)

exibir_plots(
    dados_abertura, 'Após Abertura', 
    dados_morfologia, 'Após Abertura e Fechamento'
)

""" 
=============
COMENTÁRIOS
=============
A operação de abertura não foi muito eficaz em remover pequenos ruídos presentes na imagem binária.
O fechamento foi utilizado para preencher pequenas falhas existentes no rosto e no fundo da imagem
A imagem ficou um pouco mais uniforme e adequada para a etapa de identificação. 
"""

""" 
====================
PARTES 4 E 5
====================
"""
# 4 identificação de OBJETOS
# 5 extração de CARACTERÍSTICAS

labels = label(imagem_fechamento)
propriedades = regionprops(labels)

# Lista para armazenar os dicionários de dados de cada linha da tabela
dados_tabela = []
contador_objetos_reais = 1

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(imagem, cmap='gray')

for prop in propriedades:
    if prop.area < 500: 
        continue

    area_obj = extrair_area(prop)
    perimetro_obj = extrair_perimetro(prop)
    centroide_obj = extrair_centroide(prop)
    hu1_obj = extrair_momentos_hu(prop)
    
    # Armazena p/ tabela
    dados_tabela.append({
        "Objeto ID": contador_objetos_reais,
        "Área (pixels)": area_obj,
        "Perímetro (pixels)": round(perimetro_obj, 2),
        "Centroide (X, Y)": centroide_obj,
        "Momento de Hu (h1)": f"{hu1_obj:.6e}" 
    })
    
    y0, x0 = prop.centroid
    ax.plot(x0, y0, 'ro', markersize=6)
    ax.text(x0 + 5, y0 - 5, str(contador_objetos_reais), color='yellow', fontsize=12, weight='bold')
    
    contador_objetos_reais += 1

ax.set_title("Objetos Identificados para Extração de Características")
ax.axis('off')
plt.show()

df_caracteristicas = pd.DataFrame(dados_tabela)
nome_arquivo = "extracao_caracteristicas_objetos.csv"
export_to_csv(nome_arquivo)

""" 
=============
COMENTÁRIOS
=============
Os objetos foram parcialmente identificados com sucesso, provavelmente devido aos pequenos buracos que ainda
persistiram após o fechamento morfológico.

"""

""" 
====================
PARTE 6
====================

#conclusão
É a característica mais informativa para separação de escala (macro vs. micro). 
Ela escancara que o Objeto 1 não pertence à mesma categoria dos outros dois.

Inicialmente, a imagem foi segmentada com limearização global utilizando o threshold de OTSU. Ao executar
todos os passos, os dados do CSV revelam uma falha crucial no pipeline adotado: O Cameraman não havia sido detectado como um objeto.
Para resolver esse problema, aplicamos uma inversão binária logo após a limearização.

Agora, os dados do csv estão consistentes: o objeto 1 é o homem e o objeto 2 o tripé.

"""