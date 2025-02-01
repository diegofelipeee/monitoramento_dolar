📈 Monitoramento do Dólar em Tempo Real
Este projeto é uma aplicação web desenvolvida com Flask que permite acompanhar a cotação do dólar em tempo real, analisar o sentimento das notícias relacionadas e até mesmo fazer uma previsão do câmbio usando regressão linear.

🚀 Funcionalidades
Consulta do dólar em tempo real 📊

Obtém a cotação atual diretamente do site Dólar Hoje.
Análise de sentimento de notícias 📰

Busca notícias sobre economia utilizando a API da GNews.
Analisa o sentimento (positivo, negativo ou neutro) com NLTK (VADER Sentiment Analysis).
Predição do câmbio 🔮

Utiliza regressão linear (Scikit-Learn) para tentar prever tendências futuras da cotação.
Interface Web e API 🌍

Exibe os dados de forma amigável em uma página HTML.
Oferece uma API JSON para consumo dos dados.
🛠️ Tecnologias utilizadas
Python + Flask (para a API e interface web)
BeautifulSoup + Requests (para web scraping da cotação)
NLTK (para análise de sentimento de notícias)
Scikit-Learn (Linear Regression) (para previsão de preços)
HTML + CSS (Templates Flask)
🎯 Como usar
Clone este repositório:

bash
Copiar
Editar
git clone https://github.com/seu-usuario/monitoramento-dolar.git
cd monitoramento-dolar
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
Execute a aplicação:

bash
Copiar
Editar
python api_dolar.py
Acesse no navegador:

cpp
Copiar
Editar
http://127.0.0.1:5000
📌 Observações
O projeto requer uma API Key da GNews para buscar notícias (substituir no código).
Para análise de sentimento, pode ser necessário baixar os dados do NLTK.
