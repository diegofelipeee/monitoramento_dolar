from flask import Flask, jsonify, render_template
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Baixando o recurso 'vader_lexicon' necessário para o NLTK...")
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

GNEWS_API_KEY = "Sua_API_KEY"  # Chave da GNews API
CACHE_NOTICIAS = {"data": None, "expira_em": None}

# Inicialize o analisador de sentimento
sentiment_analyzer = SentimentIntensityAnalyzer()

# Função para buscar valor do dólar em tempo real
def buscar_dolar():
    try:
        response = requests.get("https://dolarhoje.com/", timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        valor_texto = soup.find("input", {"id": "nacional"}).get("value")
        valor = float(valor_texto.replace(",", "."))
        if valor <= 0:
            raise ValueError("Valor inválido do dólar retornado")
        return valor
    except Exception as e:
        print(f"Erro ao buscar valor do dólar: {e}")
        return None

# Função para buscar notícias sobre a alta do dólar usando GNews API
def buscar_noticias_alta_dolar():
    global CACHE_NOTICIAS
    agora = datetime.now()

    if CACHE_NOTICIAS["data"] and CACHE_NOTICIAS["expira_em"] > agora:
        return CACHE_NOTICIAS["data"]

    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": "alta do dólar",
            "lang": "pt",
            "country": "br",
            "max": 5,
            "apikey": GNEWS_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        dados = response.json()

        noticias = [
            {
                "titulo": artigo.get("title", "Título não disponível"),
                "link": artigo.get("url", "#"),
                "fonte": artigo.get("source", {}).get("name", "Fonte desconhecida"),
                "data_publicacao": artigo.get("publishedAt", "Data não disponível")
            }
            for artigo in dados.get("articles", [])
        ]

        CACHE_NOTICIAS["data"] = noticias
        CACHE_NOTICIAS["expira_em"] = agora + timedelta(hours=1)
        return noticias

    except Exception as e:
        print(f"Erro ao buscar notícias da API GNews: {e}")
        return []

# Função para calcular o sentimento médio das notícias
def calcular_sentimento_noticias(noticias):
    try:
        polaridades = []
        for noticia in noticias:
            titulo = noticia.get("titulo", "")
            sentimento = sentiment_analyzer.polarity_scores(titulo)
            polaridades.append(sentimento['compound'])  # 'compound' dá o sentimento geral.

        if polaridades:
            return sum(polaridades) / len(polaridades)  # Média do sentimento.
        return 0
    except Exception as e:
        print(f"Erro ao calcular sentimento das notícias: {e}")
        return 0

# Função para buscar dados históricos do dólar
def buscar_historico_dolar(dias=30):
    try:
        url = f"https://economia.awesomeapi.com.br/json/daily/USD-BRL/{dias}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        dados = response.json()

        historico = [
            {"data": datetime.fromtimestamp(int(item["timestamp"])).strftime("%Y-%m-%d"), "valor": float(item["bid"])}
            for item in sorted(dados, key=lambda x: x["timestamp"], reverse=True)
        ]

        dias_uteis = []
        dias_vistos = set()
        for item in historico:
            data = datetime.strptime(item["data"], "%Y-%m-%d")
            if data.weekday() < 5 and item["data"] not in dias_vistos:
                dias_uteis.append(item)
                dias_vistos.add(item["data"])
            if len(dias_uteis) == 7:
                break

        return dias_uteis
    except Exception as e:
        print(f"Erro ao buscar histórico do dólar: {e}")
        return []

# Função para calcular probabilidade com base no valor do dólar em tempo real
def calcular_probabilidade_com_dolar(historico, valores_alvo, dolar_atual):
    try:
        probabilidades = {}
        valores = [item["valor"] for item in historico]

        # Verificar se temos dados históricos suficientes
        if len(valores) < 2:
            return {valor: 0 for valor in valores_alvo}

        # Cálculo da média e do desvio padrão
        media = sum(valores) / len(valores)
        desvio = (sum([(x - media) ** 2 for x in valores]) / len(valores)) ** 0.5

        # Evitar divisão por zero no cálculo do Z-score
        if desvio == 0:
            desvio = 1e-5

        # Log de depuração
        #print(f"Media: {media}, Desvio: {desvio}, Dólar Atual: {dolar_atual}")

        for alvo in valores_alvo:
            # Calcular Z-score e transformá-lo em uma probabilidade
            z = (alvo - media) / desvio
            probabilidade = max(0, min(100, round(100 - abs(z * 10), 2)))

            # Ajuste com base no dólar atual
            ajuste = 5 if dolar_atual >= alvo else -5

            # Adicionar ajuste dinâmico com base na volatilidade
            ajuste_dinamico = (abs(dolar_atual - media) / media) * 10
            probabilidade_final = round(max(0, min(100, probabilidade + ajuste + ajuste_dinamico)), 2)

            # Log de depuração para cada alvo
            #print(f"Alvo: {alvo}, Z-score: {z}, Probabilidade Base: {probabilidade}, Ajuste: {ajuste}, Ajuste Dinâmico: {ajuste_dinamico}, Probabilidade Final: {probabilidade_final}")

            # Armazenar no dicionário de probabilidades
            probabilidades[alvo] = probabilidade_final

        return probabilidades
    except Exception as e:
        print(f"Erro ao calcular probabilidades com dólar: {e}")
        return {valor: 0 for valor in valores_alvo}



# Função para calcular a variação percentual nos últimos 7 dias
def calcular_variacao_percentual(historico):
    try:
        if len(historico) >= 7:
            valor_inicial = historico[-1]["valor"]
            valor_final = historico[0]["valor"]
            variacao = ((valor_final - valor_inicial) / valor_inicial) * 100
            return round(variacao, 2)
        return 0
    except Exception as e:
        print(f"Erro ao calcular variação percentual: {e}")
        return 0

# Função para determinar o status do mercado com base em variação e notícias
def determinar_status_mercado(variacao_percentual, noticias):
    try:
        # Inicializar o analisador de sentimento
        sentiment_analyzer = SentimentIntensityAnalyzer()

        # Lista para armazenar escores de sentimento das notícias
        sentimentos = []

        # Avaliar sentimento de cada notícia
        for noticia in noticias:
            titulo = noticia.get("titulo", "")
            if titulo:  # Garantir que o título não esteja vazio
                sentimento = sentiment_analyzer.polarity_scores(titulo)
                sentimentos.append(sentimento['compound'])  # Pegar o escore "compound" (sentimento geral)

        # Calcular a média do sentimento das notícias
        sentimento_medio = sum(sentimentos) / len(sentimentos) if sentimentos else 0

        # Adicionar pesos para manchetes recentes
        pesos = [1 + (0.1 * i) for i in range(len(sentimentos))]  # Pesos crescentes para manchetes mais recentes
        sentimento_ponderado = (
            sum(s * p for s, p in zip(sentimentos, pesos)) / sum(pesos)
            if sentimentos else 0
        )

        # Conferir títulos individualmente para sentimento extremo
        sentimento_extremo = any(s < -0.5 for s in sentimentos)

        # Log para depuração
        print(f"Sentimentos das notícias: {sentimentos}")
        print(f"Sentimento médio: {sentimento_medio}, Sentimento ponderado: {sentimento_ponderado}")
        print(f"Variação percentual: {variacao_percentual}")

        # Determinar status com base em sentimento e variação
        if variacao_percentual >= 1 or sentimento_ponderado < -0.05 or sentimento_extremo:
            return "negativo"
        elif sentimento_ponderado > 0.1:
            return "positivo"
        else:
            return "estavel"
    except Exception as e:
        print(f"Erro ao determinar status do mercado: {e}")
        return "estavel"  # Retorna "estavel" como fallback





# Função para gerar previsão com base em múltiplos critérios e fundamentos das notícias
def gerar_previsao_dolar(historico, noticias, probabilidades, status_mercado, dolar_atual):
    try:
        valores = [item["valor"] for item in historico]
        datas = [i for i in range(len(valores))]
        sentimento_mercado = calcular_sentimento_noticias(noticias)
        variacao_percentual = calcular_variacao_percentual(historico)

        if len(valores) < 7:
            return "Dados insuficientes para previsão."

        modelo = LinearRegression()
        modelo.fit(np.array(datas).reshape(-1, 1), np.array(valores))

        previsoes = []
        max_valor_historico = max(valores)
        palavras_chave_positivas = ["iniciativa", "redução", "estabilidade", "planejamento"]
        palavras_chave_negativas = ["crise", "instabilidade", "inflação alta", "aumento de juros"]

        # Impacto das notícias no mercado
        impacto_governo = 0
        for noticia in noticias:
            for palavra in palavras_chave_positivas:
                if palavra.lower() in noticia["titulo"].lower():
                    impacto_governo += 0.01
            for palavra in palavras_chave_negativas:
                if palavra.lower() in noticia["titulo"].lower():
                    impacto_governo -= 0.01

        # Pesos ajustados para critérios
        peso_dolar_atual = 0.5
        peso_sentimento = 0.4
        peso_probabilidade = 0.5
        peso_variacao = 0.5
        peso_status = 0.05
        peso_noticias = 0.1

        for i in range(7):
            dia_futuro = len(datas) + i
            valor_previsto = modelo.predict([[dia_futuro]])[0]

            ajuste_dolar_atual = peso_dolar_atual * (dolar_atual - valor_previsto)
            ajuste_sentimento = peso_sentimento * sentimento_mercado
            ajuste_probabilidade = peso_probabilidade * (sum(probabilidades.values()) / len(probabilidades)) * 0.01
            ajuste_variacao = peso_variacao * variacao_percentual * 0.01
            ajuste_status = peso_status * (-1 if status_mercado == "negativo" else 1)
            ajuste_governo = peso_noticias * impacto_governo

            valor_final = (
                valor_previsto
                + ajuste_dolar_atual
                + ajuste_sentimento
                + ajuste_probabilidade
                + ajuste_variacao
                + ajuste_status
                + ajuste_governo
            )

            # Recalcular probabilidade com limite superior e inferior
            probabilidade_alvo = max(80, min(98, round((valor_final / max_valor_historico) * 100, 2)))

            previsoes.append({
                "dia": (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "valor": round(valor_final, 2),
                "probabilidade": probabilidade_alvo
            })

        return previsoes
    except Exception as e:
        print(f"Erro ao gerar previsão do dólar: {e}")
        return []

@app.route("/previsao_dolar")
def previsao_dolar():
    try:
        historico = buscar_historico_dolar(30)
        noticias = buscar_noticias_alta_dolar()
        valores_alvo = [6.30, 6.70, 7.00]
        dolar_atual = buscar_dolar()  # Certifique-se de que o valor do dólar atual é buscado aqui
        
        if dolar_atual is None:
            raise ValueError("Valor do dólar atual não disponível")

        probabilidades = calcular_probabilidade_com_dolar(historico, valores_alvo, dolar_atual)
        status_mercado = determinar_status_mercado(calcular_variacao_percentual(historico), noticias)
        
        # Passe o valor do dólar atual para a função gerar_previsao_dolar
        previsoes = gerar_previsao_dolar(historico, noticias, probabilidades, status_mercado, dolar_atual)

        return jsonify({
            "historico": historico,
            "previsoes": previsoes,
            "noticias": noticias,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"Erro ao processar previsão de dólar: {e}")
        return jsonify({
            "error": f"Erro ao processar previsão de dólar: {e}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dados")
def dados():
    try:
        dolar = buscar_dolar()
        noticias_alta_dolar = buscar_noticias_alta_dolar()
        historico = buscar_historico_dolar(30)
        valores_alvo = [6.30, 6.70, 7.00]

        probabilidades = calcular_probabilidade_com_dolar(historico, valores_alvo, dolar)
        variacao_percentual = calcular_variacao_percentual(historico)
        status_mercado = determinar_status_mercado(variacao_percentual, noticias_alta_dolar)

        return jsonify({
            "dolar": dolar,
            "noticias_alta_dolar": noticias_alta_dolar,
            "historico": historico,
            "probabilidades": probabilidades,
            "variacao_percentual": variacao_percentual,
            "status_mercado": status_mercado,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    except Exception as e:
        print(f"Erro ao gerar resposta no endpoint /dados: {e}")
        return jsonify({
            "error": "Erro ao processar os dados.",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

if __name__ == "__main__":
    app.run(debug=True)
