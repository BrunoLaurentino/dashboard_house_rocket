#COLETA DE DADOS
import pandas as pd
import streamlit as st
import numpy as np
import folium
import plotly.express as px

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster


def get_data(path):
    data = pd.read_csv(path)
    return data

def set_price(data):
    data['price_m2'] = (data['price'] / data['sqft_lot']) /10.764
    return data.copy()

def clean(data):
    #LIMPEZA DE DADOS
    data = data.drop(columns = ['sqft_living15','sqft_lot15']).copy()
    data['date'] = pd.to_datetime(data['date']).dt.strftime ('%Y-%m-%d')
    #retirando outlier (poossível erro de digitação)
    data = data[data['bedrooms'] <33]

    return data

def analyze(data):
    #mediana de quartos e de preço por região
    data2 = data[['zipcode','bedrooms','price']].groupby(['zipcode','bedrooms']).median('price').sort_values('zipcode', ascending = False).reset_index().copy()
    data2 = data2.rename(columns = {'bedrooms': 'bedrooms_median','price':'price_median'})
    #unindo dataframes
    data2 = pd.merge(data2, data, on = 'zipcode', how = 'inner')

    #filtrando tabela de mediana pela quantidade de quartos de cada imóvel / selecionando melhores imóves (condition >=2)
    data3 = data2[['id','date','zipcode','price','price_median','sqft_living','price_m2','bedrooms','yr_built','condition','lat','long']] [data2['bedrooms_median'] == data2['bedrooms']] [data2['condition'] >=2].copy()
    #selecionando colunas que farão parte da tabela final
    data3['status'] = 'NA'
    data3 = data3[['id','date','zipcode','status','price','price_median','sqft_living','price_m2','bedrooms','yr_built','lat','long']].copy().reset_index()
    data3 = data3.drop(columns = 'index')
    #análise de compra
    for i in range(len(data3)):
        if (data3.loc[i,'price'] < data3.loc[i,'price_median']):
            data3.loc[i,'status'] = 'Comprar'
        else:
            data3.loc[i,'status'] = 'Não Comprar'

    return data3

def dashboard(data,data3):
    st.set_page_config(layout= 'wide')

    #barras laterais de filtros
    st.sidebar.title('Filters')
    b_status = st.sidebar.multiselect('Enter status', sorted(set(data3['status'].unique())))
    b_zipcode = st.sidebar.multiselect('Enter zipcode', sorted(set(data3['zipcode'].unique())))

    #aplicando filtros
    if (b_zipcode != []) & (b_status != []):
        data3 =data3.loc[data3['zipcode'].isin(b_zipcode) & data3['status'].isin(b_status), : ]
    elif (b_zipcode != []) & (b_status == []):
        data3 =data3.loc[data3['zipcode'].isin(b_zipcode), :]
    elif (b_zipcode == []) & (b_status != []):
        data3 =data3.loc[data3['status'].isin(b_status), : ]
    else:
        data3 = data3.copy()

    #Tabela principal
    st.title('Data Overview')
    st.dataframe(data3,height = 400,width = 2000)
    shape = data3['id'].count()
    st.text(f'Quantidade de imóveis selecionados: {shape}')

    #average metrics
    c1,c2 = st.columns((1,1))

    df1 = data[['id','zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living','zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2','zipcode']].groupby('zipcode').mean().reset_index()
    #merge
    m1 = pd.merge(df1,df2, on= 'zipcode', how = 'inner')
    m2 = pd.merge(m1,df3, on= 'zipcode', how = 'inner')
    mt = pd.merge(m2,df4, on= 'zipcode', how = 'inner')

    mt.columns = ['ZIPCODE','TOTAL_HOUSES', 'PRICE','SQRT_LIVING', 'PRICE/M2']

    c1.subheader("Average Values")
    c1.dataframe(mt,height = 400,width = 600)

    #atributos estaticos
    n_atributes = data.select_dtypes(include=['int64','float64'])
    media = pd.DataFrame(n_atributes.apply(np.mean))
    mediana = pd.DataFrame(n_atributes.apply(np.median))
    desvio = pd.DataFrame(n_atributes.apply(np.std))
    max_ = pd.DataFrame(n_atributes.apply(np.max))
    min_ = pd.DataFrame(n_atributes.apply(np.min))

    ds = pd.concat([max_,min_,media, mediana, desvio], axis = 1).reset_index()
    ds.columns = ['attibutes', 'max', 'min', 'mean', 'median','std']

    c2.subheader("Descriptive Analysis")
    c2.dataframe(ds,height = 400,width = 600)

    #mapa
    st.title('Map')
    x1, x2 = st.columns((1,1))
    mp = data3.sample(42)
    mapa = folium.Map(location = [data3['lat'].mean(), data3['long'].mean()], defaut_zoom_start = 15)

    cluster = MarkerCluster().add_to(mapa)

    for name, row in mp.iterrows():
        folium.Marker([row['lat'], row['long']],
                  popup= 'Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, year built: {4}, status: {5}'.format(row['price'],
                                                                                                                       row['date'],
                                                                                                                       row['sqft_living'],
                                                                                                                       row['bedrooms'],
                                                                                                                       row['yr_built'],
                                                                                                                       row['status'])).add_to(cluster)

    with x1:
        folium_static(mapa)

    #season
    data['month'] = pd.to_datetime(data['date']).dt.strftime('%m')
    ds = round(data[['month','price']].groupby('month').mean('price').sort_values('month', ascending = True).reset_index(),2)
    st.subheader('Average Price per Month')
    fig = px.line(ds, x='month', y='price')
    st.plotly_chart(fig, use_container_width = True)

    st.text('Vemos um maior aumento dos preços nas estações de primavera e verão, com a maior faixa nos \nprimeiros 4 meses posteriores ao inverno. Chegando no topo em 4% de alta no mês de Abril.')
    st.header('Insights')
    st.text('Foi considerado imóveis para compra os imóveis com ótimas condições e com preço ponderado por \nquartos e zizcode a baixo da média da região.\n'
            '\nImóveis sugeridos para a compra estão em média 20% mais baratos que a média do mercado.\n'
            'Ao compararmos os imóveis com recomendação de compra com os imóveis sem recomendação verificamos\n'
            'que os anos de construção são próximos em ambos, média de 1970 contra 1976.\n'
            'A área média dos imóveis também é próxima, 2080fts contra 2079fts.\n'
            'Imóveis com vista para a água e imóveis reformados não apresentaram significância estatística\n'
            'entre os dois grupos.')

    st.subheader('Conclusão')
    st.text('Deve-se comprar imóveis listados com preço a baixo da média de sua região entre os meses\n'
            'de Dezembro à Fevereiro. E revende-los nos meses de Março à Maio.')

    st.text('----------------------------------------------------------------------------------------------\n'
            '----------------------------------------------------------------------------------------------\n'
            '----------------------------------------------------------------------------------------------')

    st.header('Other Attributes')

    #Average Price per Year
    st.subheader('Average Price per Year')
    dzz = data3[['yr_built','price']].groupby('yr_built').mean().reset_index()
    fig = px.line(dzz, x = 'yr_built', y = 'price')
    st.plotly_chart( fig, use_container_width = True)

    #Average Price per Day
    st.subheader('Average Price per Day')
    dzz = data3[['date','price']].groupby('date').mean().reset_index()
    fig = px.line(dzz, x = 'date', y = 'price')
    st.plotly_chart( fig, use_container_width = True)

    #histograma
    x1,x2 = st.columns((1,1))

    x1.subheader('Price Distribution')
    fig = px.histogram(data3, x = 'price', nbins =50)
    x1.plotly_chart( fig, use_container_width = True)

    x2.subheader('Bedrooms Distribution')
    fig = px.histogram(data3, x = 'bedrooms')
    x2.plotly_chart( fig, use_container_width = True)

    url = "https://brunolaurentino.github.io/portfolio_projetos/"
    st.markdown("Check my page [link](%s)" % url)


if __name__ == '__main__':
    #ETL
    #Extration
    path = 'kc_house_data.csv'
    df = get_data(path)
    #Transformation
    df = set_price(df)
    df = clean(df)
    dz = analyze(df)
    #Loading
    dashboard(df,dz)
