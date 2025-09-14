import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.regression import load_model, predict_model

# --- CSS ---
st.markdown(
    """
    <style>
    .stApp { background-color: black; color: #ffffff; }
    .stButton>button { background-color: #6B5B95; color: white; border-radius: 8px; }
    .stSidebar { background-color: black; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Carregando modelo ---
modelo = load_model('modelo_final_vendas')

# --- Título ---
st.markdown('<h3>Painel de Análise de Vendas de Jogos</h3>', unsafe_allow_html=True)

# --- Dataset padrão ---
novos_dados = pd.read_csv('vgsales.csv')

# --- Upload opcional ---
uploaded_file = st.file_uploader("Carregar CSV de Vendas (opcional)", type="csv")
if uploaded_file:
    novos_dados = pd.read_csv(uploaded_file)
    st.success("Novo arquivo carregado!")
else:
    st.info("Usando dataset padrão.")

st.dataframe(novos_dados)

# --- Previsões ---
previsoes = predict_model(modelo, data=novos_dados)
previsoes_ext = previsoes.copy()
for col in ['Name', 'Year', 'Platform', 'Genre', 'Publisher']:
    if col in novos_dados.columns:
        previsoes_ext[col] = novos_dados[col]

# --- Filtros ---
st.sidebar.header("Filtros")
plataformas = ['All'] + list(previsoes_ext['Platform'].dropna().unique()) if 'Platform' in previsoes_ext.columns else []
generos = ['All'] + list(previsoes_ext['Genre'].dropna().unique()) if 'Genre' in previsoes_ext.columns else []
publicadoras = ['All'] + list(previsoes_ext['Publisher'].dropna().unique()) if 'Publisher' in previsoes_ext.columns else []

plataforma_sel = st.sidebar.multiselect("Plataformas:", plataformas, default=['All'])
genero_sel = st.sidebar.multiselect("Gêneros:", generos, default=['All'])
publisher_sel = st.sidebar.multiselect("Publicadoras:", publicadoras, default=['All'])

filtrado = previsoes_ext.copy()
if 'Platform' in filtrado.columns and 'All' not in plataforma_sel:
    filtrado = filtrado[filtrado['Platform'].isin(plataforma_sel)]
if 'Genre' in filtrado.columns and 'All' not in genero_sel:
    filtrado = filtrado[filtrado['Genre'].isin(genero_sel)]
if 'Publisher' in filtrado.columns and 'All' not in publisher_sel:
    filtrado = filtrado[filtrado['Publisher'].isin(publisher_sel)]

# --- Seleção de regiões ---
regioes = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
regioes_sel = st.sidebar.multiselect("Regiões:", regioes, default=regioes)
vendas_total = filtrado[regioes_sel].sum().sum() if len(regioes_sel) > 0 else 0

# --- Indicadores ---
st.markdown('<h3>Indicadores Principais</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric(f"Total de Vendas ({', '.join(regioes_sel)})", round(vendas_total, 2))
col2.metric("Maior Venda", round(filtrado['Global_Sales'].max(), 2) if not filtrado.empty else 0)
col3.metric("Quantidade de Jogos", len(filtrado))

# --- Abas ---
tab1, tab2, tab3, tab4 = st.tabs([" Vendas por Ano", " Top Jogos", " Plataformas & Distribuições", "Análises avançadas"])

# --- Vendas por Ano ---
with tab1:
    if not filtrado.empty and 'Year' in filtrado.columns:
        vendas_ano = filtrado.groupby('Year')['Global_Sales'].sum().reset_index()
        fig_ano = px.line(vendas_ano, x='Year', y='Global_Sales', title="Vendas Totais por Ano",
                          labels={'Year':'Ano','Global_Sales':'Vendas Globais'}, markers=True)
        st.plotly_chart(fig_ano, use_container_width=True)

        if len(regioes_sel) > 0:
            vendas_ano_regiao = filtrado.groupby('Year')[regioes_sel].sum().reset_index()
            fig_regiao = px.bar(vendas_ano_regiao, x='Year', y=regioes_sel,
                                title="Vendas por Ano por Região", labels={'value':'Vendas','Year':'Ano'}, text_auto=True)
            st.plotly_chart(fig_regiao, use_container_width=True)
    else:
        st.info("Nenhum dado disponível para Vendas por Ano.")

# --- Top Jogos ---
with tab2:
    if not filtrado.empty and 'Name' in filtrado.columns:
        top_jogos = filtrado.sort_values(by='Global_Sales', ascending=False).head(10)
        if not top_jogos.empty:
            fig_top = px.bar(top_jogos, x='Name', y='Global_Sales', title="Top 10 Jogos com Maiores Vendas",
                             labels={'Name':'Nome do Jogo','Global_Sales':'Vendas Globais'},
                             text='Global_Sales', color='Global_Sales', color_continuous_scale=px.colors.sequential.Viridis)
            fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_top.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top, use_container_width=True)

            fig_treemap = px.treemap(top_jogos, path=['Genre','Name'], values='Global_Sales',
                                     title="Top Jogos por Gênero")
            st.plotly_chart(fig_treemap, use_container_width=True)
        else:
            st.info("Nenhum jogo corresponde aos filtros selecionados.")
    else:
        st.info("Nenhum dado disponível para Top Jogos.")

# --- Plataformas & Distribuições ---
with tab3:
    col1, col2 = st.columns(2)
    if not filtrado.empty and 'Platform' in filtrado.columns:
        vendas_plat = filtrado.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).reset_index()
        fig_plat = px.bar(vendas_plat, x='Platform', y='Global_Sales',
                          title="Vendas por Plataforma",
                          labels={'Platform':'Plataforma','Global_Sales':'Vendas Globais'},
                          text_auto=True, color='Global_Sales', color_continuous_scale=px.colors.sequential.Plasma)
        col1.plotly_chart(fig_plat, use_container_width=True)

    if not filtrado.empty and len(regioes_sel) > 0:
        vendas_paises = pd.DataFrame({'Região': regioes_sel, 'Vendas': [filtrado[r].sum() for r in regioes_sel]})
        fig_donut = px.pie(vendas_paises, names='Região', values='Vendas',
                           title='Distribuição de Vendas por Região', hole=0.4,
                           color_discrete_sequence=px.colors.sequential.Viridis[:len(regioes_sel)])
        fig_donut.update_traces(textinfo='percent+label', textfont=dict(color='white', size=14))
        col2.plotly_chart(fig_donut, use_container_width=True)

    if not filtrado.empty and 'Genre' in filtrado.columns:
        vendas_genero = filtrado.groupby('Genre')['Global_Sales'].sum().reset_index()
        fig_genero = px.pie(vendas_genero, names='Genre', values='Global_Sales',
                            title='Distribuição de Vendas por Gênero', hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_genero.update_traces(textinfo='percent+label', textfont=dict(color='white', size=12))
        st.plotly_chart(fig_genero, use_container_width=True)

        # Linha de tendência por gênero
        genero_year = filtrado.groupby(['Year','Genre'])['Global_Sales'].sum().reset_index()
        fig_gen = px.line(genero_year, x="Year", y="Global_Sales", color="Genre",
                          title="Evolução das Vendas por Gênero", markers=True)
        st.plotly_chart(fig_gen, use_container_width=True)

# --- Análises Avançadas ---
with tab4:
    if not filtrado.empty and 'Year' in filtrado.columns and 'Genre' in filtrado.columns:
        heatmap_data = filtrado.groupby(['Year','Genre'])['Global_Sales'].sum().reset_index()
        fig_heatmap = px.density_heatmap(heatmap_data, x="Year", y="Genre", z="Global_Sales",
                                         color_continuous_scale="hot",
                                         title="Heatmap de Vendas por Ano e Gênero")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Nenhum dado disponível para o Heatmap.")

    if not filtrado.empty and 'Publisher' in filtrado.columns:
        top_pub = filtrado.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        if not top_pub.empty:
            fig_pub = px.bar(top_pub, x="Global_Sales", y="Publisher", orientation="h",
                             title="Top 10 Publicadoras por Vendas Globais",
                             labels={"Global_Sales":"Vendas Globais","Publisher":"Publicadora"},
                             text_auto=True, color="Global_Sales", color_continuous_scale=px.colors.sequential.Magma)
            st.plotly_chart(fig_pub, use_container_width=True)
        else:
            st.info("Nenhuma publicadora corresponde aos filtros selecionados.")
    else:
        st.info("Nenhum dado disponível para Top Publicadoras.")

    if not filtrado.empty and 'Year' in filtrado.columns and 'Platform' in filtrado.columns:
        plat_year = filtrado.groupby(['Year','Platform'])['Global_Sales'].sum().reset_index()
        fig_area = px.area(plat_year, x="Year", y="Global_Sales", color="Platform",
                           title="Evolução das Plataformas ao Longo do Tempo",
                           labels={"Year":"Ano","Global_Sales":"Vendas Globais"})
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("Nenhum dado disponível para Evolução das Plataformas.")

