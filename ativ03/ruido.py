img1_path=r"C:\Users\LENOVO\Documents\VisaoComputacional\atividades\ativ03\img\banco2.jpg"
img2_path=r"C:\Users\LENOVO\Documents\VisaoComputacional\atividades\ativ03\img\sapato1.jpg"

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_cv2(caminho_imagem):
    img = cv2.imread(caminho_imagem, 0)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_imagem}")
    return img

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

    """
    Aplica o Filtro Passa-Baixa Gaussiano (GLPF) no domínio da frequência
    e retorna a imagem filtrada de volta ao domínio espacial com suas métricas.
    """
    rows, cols = img_ruidosa.shape
    crow, ccol = rows // 2, cols // 2

    x, y = np.ogrid[:rows, :cols]
    dist_quadrada = (x - crow)**2 + (y - ccol)**2
    mask = np.exp(-dist_quadrada / (2 * (radius ** 2)))

    f_transform = np.fft.fft2(img_ruidosa)
    f_shift = np.fft.fftshift(f_transform)
    filtered = f_shift * mask

    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return {
        "imagem_filtrada": img_back,
        "mascara_frequencia": mask,
        "media": np.mean(img_back),
        "variancia": np.var(img_back)
    }

    """
    Aplica o Filtro Passa-Alta Gaussiano (GHPF) no domínio da frequência
    e retorna a imagem de altas frequências (bordas) de volta ao domínio espacial.
    """
    rows, cols = img_ruidosa.shape
    crow, ccol = rows // 2, cols // 2

    x, y = np.ogrid[:rows, :cols]
    dist_quadrada = (x - crow)**2 + (y - ccol)**2
    
    # EQUAÇÃO DO PASSA-ALTA: 1 - exp(-D^2 / 2D0^2)
    # Nota: para passa-alta, raios menores (ex: 15 a 30) costumam preservar mais a estrutura
    mask = 1 - np.exp(-dist_quadrada / (2 * (radius ** 2)))

    f_transform = np.fft.fft2(img_ruidosa)
    f_shift = np.fft.fftshift(f_transform)
    filtered = f_shift * mask

    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return {
        "imagem_filtrada": img_back,
        "mascara_frequencia": mask
    }

def suavizacao_gaussiana_pequena(img_ruidosa):
    """Aplica filtro Gaussiano com kernel pequeno 3x3 (Suavização Leve)."""
    img_filtrada = cv2.GaussianBlur(img_ruidosa, (3, 3), 0)
    return {"imagem_filtrada": img_filtrada}

def suavizacao_gaussiana_grande(img_ruidosa):
    """Aplica filtro Gaussiano com kernel maior 7x7 (Suavização Intensa)."""
    img_filtrada = cv2.GaussianBlur(img_ruidosa, (7, 7), 0)
    return {"imagem_filtrada": img_filtrada}

def calcular_psnr(img_limpa, img_filtrada):
    """
    Calcula o PSNR (Peak Signal-to-Noise Ratio) entre a imagem original limpa
    e a imagem após passar pelo filtro de suavização.
    """
    # Garante que as imagens estão no formato float para não estourar os limites do uint8
    mse = np.mean((img_limpa.astype(np.float64) - img_filtrada.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def passa_alta_laplaciano(img_ruidosa):
    """
    Aplica o Filtro Passa-Alta Laplaciano no domínio da frequência
    e retorna o mapa de bordas de segunda derivada no domínio espacial.
    """
    rows, cols = img_ruidosa.shape
    crow, ccol = rows // 2, cols // 2

    x, y = np.ogrid[:rows, :cols]
    dist_quadrada = (x - crow)**2 + (y - ccol)**2
    
    max_dist = np.max(dist_quadrada)
    mask = dist_quadrada / max_dist
    f_transform = np.fft.fft2(img_ruidosa)
    f_shift = np.fft.fftshift(f_transform)
    filtered = f_shift * mask

    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)

    return {
        "imagem_filtrada": img_back,
        "mascara_frequencia": mask
    }

def exibir_plots_psnr(img1_k3, psnr1_k3, img1_k7, psnr1_k7, nome_img1,
                                 img2_k3, psnr2_k3, img2_k7, psnr2_k7, nome_img2):
    """
    Gera uma matriz de subplots 2x2 para comparar visualmente e quantitativamente (via PSNR)
    os efeitos dos kernels 3x3 e 7x7 em duas imagens distintas.
    """
    fig, eixos = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Análise Comparativa de Suavização Gaussiana via PSNR", fontsize=16, fontweight='bold')

    # ---- LINHA 0: Primeira Imagem (Ex: Banco) ----
    # Coluna 0: Kernel 3x3
    eixos[0, 0].imshow(img1_k3, cmap='gray')
    eixos[0, 0].set_title(f"{nome_img1} - Kernel 3x3\nPSNR: {psnr1_k3:.2f} dB")
    eixos[0, 0].axis('off')

    # Coluna 1: Kernel 7x7
    eixos[0, 1].imshow(img1_k7, cmap='gray')
    eixos[0, 1].set_title(f"{nome_img1} - Kernel 7x7\nPSNR: {psnr1_k7:.2f} dB")
    eixos[0, 1].axis('off')

    # ---- LINHA 1: Segunda Imagem (Ex: Sapato) ----
    # Coluna 0: Kernel 3x3
    eixos[1, 0].imshow(img2_k3, cmap='gray')
    eixos[1, 0].set_title(f"{nome_img2} - Kernel 3x3\nPSNR: {psnr2_k3:.2f} dB")
    eixos[1, 0].axis('off')

    # Coluna 1: Kernel 7x7
    eixos[1, 1].imshow(img2_k7, cmap='gray')
    eixos[1, 1].set_title(f"{nome_img2} - Kernel 7x7\nPSNR: {psnr2_k7:.2f} dB")
    eixos[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def salvar_metricas_csv(dados_img1, nome_base_img1, psnr_img1,
                        dados_img2, nome_base_img2, psnr_img2,
                        nome_pasta, sufixo_arquivo):
    """
    Salva as métricas estatísticas (Média, Variância e PSNR) de duas imagens
    em arquivos .csv separados, organizados dentro de uma pasta específica.
    """
    if not os.path.exists(nome_pasta):
        os.makedirs(nome_pasta)
        print(f"=== [DEBUG] === Pasta '{nome_pasta}' criada.")

    caminho_csv1 = os.path.join(nome_pasta, f"{nome_base_img1}_{sufixo_arquivo}.csv")
    with open(caminho_csv1, mode='w', newline='', encoding='utf-8') as f1:
        writer = csv.writer(f1)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Média", f"{dados_img1['media']:.4f}"])
        writer.writerow(["Variância", f"{dados_img1['variancia']:.4f}"])
        writer.writerow(["PSNR (dB)", f"{psnr_img1:.2f}" if psnr_img1 is not None else "N/A"])
    
    caminho_csv2 = os.path.join(nome_pasta, f"{nome_base_img2}_{sufixo_arquivo}.csv")
    with open(caminho_csv2, mode='w', newline='', encoding='utf-8') as f2:
        writer = csv.writer(f2)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Média", f"{dados_img2['media']:.4f}"])
        writer.writerow(["Variância", f"{dados_img2['variancia']:.4f}"])
        writer.writerow(["PSNR (dB)", f"{psnr_img2:.2f}" if psnr_img2 is not None else "N/A"])

    print(f"[DEBUG] Dados exportados: {caminho_csv1} e {caminho_csv2}")

def tabela_final():
    """
    Varre os diretórios do projeto, lê os ficheiros .csv individuais de métricas,
    e consolida tudo numa única tabela comparativa final completa.
    """
    linhas_tabela = []

    mapeamento_pastas = {
        'filtro_sal_e_pimenta': 'Ruído Sal e Pimenta',
        'filtro_gaussiano': 'Ruído Gaussiano',
        'filtro_passa_baixa_gaussiano_': 'Filtro Passa Baixa Gaussiano',
        'filtro_passa_alta_laplace_': 'Passa-Alta Laplaciano' 
    }

    for pasta, categoria in mapeamento_pastas.items():
        if not os.path.exists(pasta):
            print(f"Aviso: A pasta '{pasta}' ainda não existe ou não foi processada.")
            continue
            
        for arquivo in os.listdir(pasta):
            if arquivo.endswith('.csv'):
                caminho_completo = os.path.join(pasta, arquivo)

                objeto = 'Banco' if arquivo.lower().startswith('banco') else 'Sapato'
                configuracao = arquivo.replace(f"{objeto.lower()}_", "").replace(".csv", "").replace("_", " ").title()
                media, variancia, psnr = None, None, None

                try:
                    with open(caminho_completo, mode='r', encoding='utf-8') as f:
                        next(f) # Pula o cabeçalho "Métrica,Valor"
                        for linha in f:
                            partes = linha.strip().split(',')
                            if len(partes) < 2:
                                continue
                            metrica, valor = partes[0], partes[1]
                            
                            if 'Média' in metrica:
                                media = float(valor)
                            elif 'Variância' in metrica:
                                variancia = float(valor)
                            elif 'PSNR' in metrica:
                                psnr = valor
                except Exception as e:
                    print(f"Erro ao ler o ficheiro {arquivo}: {e}")
                    continue

                linhas_tabela.append({
                    "Imagem": objeto,
                    "Categoria": categoria,
                    "Configuração / Filtro": configuracao,
                    "Média": media,
                    "Variância": variancia,
                    "PSNR (dB)": psnr
                })

    if not linhas_tabela:
        print("Nenhum dado foi encontrado para consolidar!")
        return None

    # Transforma em DataFrame e ordena para ficar legível
    df_final = pd.DataFrame(linhas_tabela)
    df_final = df_final.sort_values(by=["Imagem", "Categoria", "Configuração / Filtro"]).reset_index(drop=True)

    df_final.to_csv("tabela_comparativa_filtros.csv", index=False, encoding='utf-8')
    
    try:
        df_final.to_excel("tabela_comparativa_filtros.xlsx", index=False)
        print("[DEBUG] Tabela final atualizada com SUCESSO tanto em .csv quanto em .xlsx!")
    except ImportError:
        print("[INFO] Tabela final atualizada com SUCESSO em 'tabela_comparativa_filtros.csv'!")

    return df_final


""" 
================
    PARTE 1 
================  
"""
#importando as imagens e extraindo caracteristicas iniciais

img1=parse_cv2(img1_path)
img2=parse_cv2(img2_path)

dados_banco = calcular_metricas_imagem(img1_path)
dados_sapato = calcular_metricas_imagem(img2_path)

# Testando plots
extrair_info(img1_path)
extrair_info(img2_path)
exibir_plots(dados_banco,'Banco',dados_sapato,'Sapato') #imagem original

####

""" 
================
    PARTE 2 
================   
"""
#aplicando ruidos

#################
# 2.1. ruido GAUSSIANO

noise1 = np.random.normal(0, 30, img1.shape)
noise2=np.random.normal(0,30,img2.shape)

img1_ruido = np.clip(img1 + noise1, 0, 255).astype(np.uint8)
img2_ruido = np.clip(img2 + noise2, 0, 255).astype(np.uint8)

dados_banco_ruido = calcular_metricas_imagem(img1_ruido)
dados_sapato_ruido = calcular_metricas_imagem(img2_ruido)

psnr_banco_ruido=calcular_psnr(img1,img1_ruido)
psnr_sapato_ruido=calcular_psnr(img2,img2_ruido)

exibir_plots(dados_banco_ruido, 'Banco com Ruído Gauss', dados_sapato_ruido, 'Sapato com Ruído Gauss') #imagem c ruido gaussiano
salvar_metricas_csv (
    dados_banco_ruido, 'banco', psnr_banco_ruido,
    dados_sapato_ruido, 'sapato', psnr_sapato_ruido,
    nome_pasta='filtro_gaussiano',
    sufixo_arquivo='ruido_gauss'
)
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

#################
# 2.2. ruido SAL E PIMENTA

img1_sp = salt_and_pepper_noise(img1, 0.2)
img2_sp = salt_and_pepper_noise(img2, 0.2)

dados_banco_sp = calcular_metricas_imagem(img1_sp)
dados_sapato_sp = calcular_metricas_imagem(img2_sp)

psnr_banco_sp=calcular_psnr(img1,img1_sp)
psnr_sapato_sp=calcular_psnr(img2,img2_sp)

exibir_plots(dados_banco_sp,'Banco ruído sp',dados_sapato_sp,'Sapato ruído sp') #imagem com ruido sal e pimenta
salvar_metricas_csv (
    dados_banco_sp, 'banco', psnr_banco_sp,
    dados_sapato_sp, 'sapato', psnr_sapato_sp,
    nome_pasta='filtro_sal_e_pimenta',
    sufixo_arquivo='ruido_sp'
)
""" 
=================== 
PARTE 2: COMENTARIOS 
===================
*porcentagem de pixels alterados *: 20%

1-Como esse ruído se diferencia visualmente do ruído gaussiano?: 
O ruído de sal e pimenta altera apenas uma porcentagem isolada de pixels específicos, deixando um aspecto salpicado
na imagem por forçar os pixels a assumirem valores extremos (0 ou 255). Os pixels vizinhos não afetados continuam limpos.
Eis a principal diferença para o ruído gaussiano, que gera uma perturbação em TODA a extensão da imagem.

Em relação às imagens desse notebook, o sal e pimenta causa uma alteração mais significativa na variância e no histograma das imagens
que o ruído gaussiano, pois o sp não trabalha com "faixas de transição" de intensidade. Isso também impacta na variância, pois
o pixel sai do valor médio para assumir algum extremo (0 ou 255), intensificando a dispersão.

2-Em quais situações esse ruído aparece em sistemas reais de captura de imagem?: 
Geralmente, aparece devido a erros na transmissão de dados, falhas de célula de memória ou até mesmo 
na conversão analógico-digital.
"""

####

""" 
================
    PARTE 3 
================  
"""
#aplicando filtros

#################
# 3.1. filtro de PASSA BAIXA GAUSSIANO

# A. teste com kernel de 3x3 (sauve)
banco_suave_3x3=suavizacao_gaussiana_pequena(img1_ruido)
sapato_suave_3x3=suavizacao_gaussiana_pequena(img2_ruido)
dados_banco_3x3 = calcular_metricas_imagem(banco_suave_3x3["imagem_filtrada"])
dados_sapato_3x3= calcular_metricas_imagem(sapato_suave_3x3["imagem_filtrada"])

exibir_plots(dados_banco_3x3, 'Banco P.B. Gauss 3x3', dados_sapato_3x3, 'Sapato P.B Gauss Kernel 3x3')


# B. teste com kernel de 7x7 (+intenso)
banco_suave_7x7 = suavizacao_gaussiana_grande(img1_ruido)
sapato_suave_7x7 = suavizacao_gaussiana_grande(img2_ruido)
dados_banco_7x7 = calcular_metricas_imagem(banco_suave_7x7["imagem_filtrada"])
dados_sapato_7x7 = calcular_metricas_imagem(sapato_suave_7x7["imagem_filtrada"])

exibir_plots(dados_banco_7x7, 'Banco P.B Gauss Kernel 7x7', dados_sapato_7x7, 'Sapato P.B Gauss Kernel 7x7')

#comparando qual kernel foi melhor: PSNR como métrica principal de análise
psnr_banco_3x3 = calcular_psnr(img1, banco_suave_3x3["imagem_filtrada"])
psnr_banco_7x7 = calcular_psnr(img1, banco_suave_7x7["imagem_filtrada"])
psnr_sapato_3x3 = calcular_psnr(img2, sapato_suave_3x3["imagem_filtrada"])
psnr_sapato_7x7 = calcular_psnr(img2, sapato_suave_7x7["imagem_filtrada"])

""" exibir_plots_psnr (
    banco_suave_3x3["imagem_filtrada"], psnr_banco_3x3, 
    banco_suave_7x7["imagem_filtrada"], psnr_banco_7x7, 
    'Banco',
    sapato_suave_3x3["imagem_filtrada"], psnr_sapato_3x3, 
    sapato_suave_7x7["imagem_filtrada"], psnr_sapato_7x7, 
    'Sapato'
) """

salvar_metricas_csv (
    dados_banco_3x3, 'banco', psnr_banco_3x3,
    dados_sapato_3x3, 'sapato', psnr_sapato_3x3,
    nome_pasta='filtro_passa_baixa_gaussiano_',
    sufixo_arquivo='3x3'
)

salvar_metricas_csv (
    dados_banco_7x7, 'banco', psnr_banco_7x7,
    dados_sapato_7x7, 'sapato', psnr_sapato_7x7,
    nome_pasta='filtro_passa_baixa_gaussiano_',
    sufixo_arquivo='7x7'
)

""" 
=================== 
PARTE 3: COMENTARIOS 
===================
Os resultados quantitativos apontaram um PSNR superior para o Filtro Gaussiano com kernel 7x7 em comparação ao kernel 3x3.
"""

####

""" 
================
    PARTE 4 
================  
"""
#filtro de passa alta

#apliquei na imagem original porque, quando apliquei na img com ruido gaussiano, ficou completamente ilegivel devido aos componentes de alta frequencia
#banco_pa_laplace = passa_alta_laplaciano(img1_ruido) 
#sapato_pa_laplace = passa_alta_laplaciano(img2_ruido)

banco_pa_laplace = passa_alta_laplaciano(img1) 
sapato_pa_laplace = passa_alta_laplaciano(img2)

dados_banco_pa_laplace = calcular_metricas_imagem(banco_pa_laplace["imagem_filtrada"])
dados_sapato_pa_laplace = calcular_metricas_imagem(sapato_pa_laplace["imagem_filtrada"])

banco_laplace_psnr=calcular_psnr(img1,banco_pa_laplace["imagem_filtrada"])
sapato_laplace_psnr=calcular_psnr(img2,sapato_pa_laplace["imagem_filtrada"])

exibir_plots(dados_banco_pa_laplace, 'Banco Passa Alta Laplace', dados_sapato_pa_laplace, 'Sapato Passa Alta Laplace')
salvar_metricas_csv (
    dados_banco_pa_laplace, 'banco_laplace',banco_laplace_psnr,
    dados_sapato_pa_laplace, 'sapato_laplace', sapato_laplace_psnr,
    nome_pasta='filtro_passa_alta_laplace_',
    sufixo_arquivo='_passaalta'
)
""" 
=================== 
PARTE 4: COMENTARIOS 
===================
Quando aplicado na imagem com ruído, o resultado do filtro apresentou degradação significativa. Como o ruído Gaussiano 
inserido no passo anterior possui alta frequência espacial (variações bruscas pixel a pixel) e o filtro Laplaciano é 
sensível a altas frequências devido à sua natureza derivativa, ele acabou amplificando o ruído em vez de 
destacar apenas as bordas estruturais do banco e do sapato. Quando aplicado na imagem original, sem o ruído, o filtro conseguiu demarcar melhor algumas bordas.
As respostas abaixo serão dadas em relação à aplicação do filtro na imagem original (sem ruído)

1-que estruturas da imagem foram destacadas?: componentes de alta frequência espacial
2-as bordas ficaram mais visíveis?: sim, especialmente os rejuntes da imagem do banco
3-Esse tipo de filtro ajudaria em quais tarefas de visão computacional?: detecção de bordas e segmentação de fundo, 
detecção de foco automático, identificação de pontos de interesse (como quinas) etc

"""

####

""" 
================
    PARTE 5 
================  
"""
#Tabela
""" Responder: 
 Qual ruído degradou mais a imagem? Laplaciano.
 O filtro gaussiano conseguiu recuperar a qualidade visual? Não muito.  
 Em quais aplicações de visão computacional seria importante aplicar filtros antes do processamento?
os dados brutos capturados por sensores de câmeras raramente estão prontos para os algoritmos de alto nível. 
Aplicar filtros no pré-processamento é uma etapa indispensável na grande maioria dos sistemas em produção. 
Em sistemas que precisam identificar caracteres em movimento (como radares de trânsito), as imagens sofrem 
com ruído proveniente de baixa luminosidade ou desfoque de movimento (motion blur). Para solucionar o problema,
Filtros Gaussianos ou de Mediana são utilizados para eliminar a granulação do sensor antes de passar a imagem para o OCR.
Sem isso, a leitura de placas seria degradada
 """

if __name__ == "__main__":
    tabela_completa = tabela_final()
