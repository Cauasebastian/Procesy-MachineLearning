import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, jsonify


def gerar_dados_aleatorios(num_registros=100000):
    tipos_processo = ["Ação Trabalhista", "Ação Cível", "Penal"]
    tipos_atendimento = ["Online", "Presencial"]
    status = ["Em Andamento", "Aguardando", "Concluído"]
    status_documentos = ["Completo", "Pendente"]
    
    data = []

    for i in range(num_registros):
        tipo_processo = random.choice(tipos_processo)
        tipo_atendimento = random.choice(tipos_atendimento)
        status_atual = random.choice(status)
        
        # Geração de tempos aleatórios
        tempo_inicio = random.randint(30, 400)
        tempo_atualizacao = random.randint(1, 60)
        
        # Status de documentos com maior chance de "Pendente"
        status_contrato = random.choices(status_documentos, weights=[0.3, 0.7])[0]
        status_proc = random.choices(status_documentos, weights=[0.4, 0.6])[0]
        status_peticao = random.choices(status_documentos, weights=[0.5, 0.5])[0]
        status_doc_complementar = random.choices(status_documentos, weights=[0.3, 0.7])[0]
        
        # Lógica para influenciar o tempo de resposta baseado nos status dos documentos
        tempo_resposta = random.randint(30, 100)

        if status_contrato == "Pendente":
            tempo_resposta += random.randint(20, 50)  # Atraso adicional para contrato pendente
        if status_proc == "Pendente":
            tempo_resposta += random.randint(10, 40)  # Atraso adicional para procuração pendente
        if status_peticao == "Pendente":
            tempo_resposta += random.randint(5, 30)   # Atraso adicional para petição pendente
        if status_doc_complementar == "Pendente":
            tempo_resposta += random.randint(10, 50)  # Atraso adicional para documento complementar pendente
        
        # Adiciona um fator de tempo com base no tipo de processo (Processos mais complexos podem levar mais tempo)
        if tipo_processo == "Ação Trabalhista":
            tempo_resposta += random.randint(20, 40)  # Ações Trabalhistas podem ser mais demoradas
        elif tipo_processo == "Ação Cível":
            tempo_resposta += random.randint(10, 30)  # Ações Cíveis tendem a ser mais rápidas

        # Adiciona um fator de tempo baseado no tipo de atendimento (Online pode ser mais rápido que Presencial)
        if tipo_atendimento == "Presencial":
            tempo_resposta += random.randint(5, 20)  # Atendimento presencial pode aumentar o tempo de resposta
        
        data.append([i+1, tipo_processo, tipo_atendimento, status_atual, tempo_inicio, tempo_atualizacao,
                    status_contrato, status_proc, status_peticao, status_doc_complementar, tempo_resposta])

    df = pd.DataFrame(data, columns=["ID", "Tipo de Processo", "Tipo de Atendimento", "Status", "Tempo de Início (dias)",
                                    "Tempo de Atualização (dias)", "Status Contrato", "Status Procuração",
                                    "Status Petição Inicial", "Status Documento Complementar", "Tempo de Resposta (dias)"])
    
    # Dowload the dataset to a CSV file
    df.to_csv('dados_processo.csv', index=False)
    print("Dados gerados e salvos em 'dados_processo.csv'")
    return df

# 2. Pré-processamento e treinamento do modelo
def treinar_modelo(df):
    # Criar cópia para não alterar o original
    df_processed = df.copy()
    
    # Inicializar e ajustar LabelEncoders
    label_encoders = {}
    categorical_cols = [
        'Tipo de Processo', 
        'Tipo de Atendimento', 
        'Status',
        'Status Contrato', 
        'Status Procuração', 
        'Status Petição Inicial', 
        'Status Documento Complementar'
    ]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Separar features e target
    X = df_processed.drop(columns=['Tempo de Resposta (dias)', 'ID'])
    y = df_processed['Tempo de Resposta (dias)']
    
    # Normalizar features numéricas
    scaler = StandardScaler()
    numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)']
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo XGBoost
    model = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE do modelo XGBoost: {mae:.2f} dias")

    # Plotar as predições versus valores reais
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # linha ideal
    plt.title('Predições vs. Reais')
    plt.xlabel('Valor Real')
    plt.ylabel('Predição')
    plt.show()
    
    return model, label_encoders, scaler, X.columns.tolist()

# 3. Configurar e iniciar a API Flask
app = Flask(__name__)

# Variáveis globais para armazenar o modelo e pré-processadores
model = None
label_encoders = None
scaler = None
colunas = None

@app.route('/prever-tempo-resposta', methods=['POST'])
def prever_tempo_resposta():
    global model, label_encoders, scaler, colunas
    
    if model is None:
        return jsonify({"erro": "Modelo não carregado"}), 500
    
    try:
        # Receber dados
        dados = request.json
        
        # Verificar formato
        if not isinstance(dados, list):
            return jsonify({"erro": "Formato inválido. Esperado lista de registros"}), 400
        
        # Criar DataFrame
        df_input = pd.DataFrame(dados)
        
        # Verificar colunas
        if not all(col in df_input.columns for col in colunas):
            return jsonify({"erro": f"Colunas faltando. Esperadas: {colunas}"}), 400
        
        # Manter apenas colunas necessárias na ordem correta
        df_input = df_input[colunas]
        
        # Aplicar pré-processamento
        df_processed = df_input.copy()
        for col, encoder in label_encoders.items():
            # Tratar valores desconhecidos (usar mais frequente)
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df_processed[col] = encoder.transform(df_processed[col])
        
        # Normalizar numéricas
        numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)']
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
        
        # Fazer previsão
        predicoes = model.predict(df_processed)
        
        # Formatar resposta
        resposta = [{"tempo_resposta_dias": float(pred)} for pred in predicoes]
        
        return jsonify(resposta)
    
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/')
def home():
    return "API de Previsão de Tempo de Resposta - Envie POST para /prever-tempo-resposta"

def inicializar_aplicacao():
    global model, label_encoders, scaler, colunas
    
    print("Gerando dados e treinando modelo...")
    df = gerar_dados_aleatorios(1000)
    
    print("Treinando modelo...")
    model, label_encoders, scaler, colunas = treinar_modelo(df)
    
    print("Modelo treinado e API pronta para receber requisições")

if __name__ == '__main__':
    # Inicializar o modelo antes de iniciar o servidor
    inicializar_aplicacao()
    
    # Iniciar servidor Flask
    print("Iniciando servidor Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
