import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, jsonify

# Constants
MODEL_FILE = 'modelo_tempo_resposta.pkl'
PREPROCESSORS_FILE = 'preprocessors.pkl'

app = Flask(__name__)

# Global variables for model and preprocessors
model = None
label_encoders = None
scaler = None
colunas = None

def gerar_dados_aleatorios(num_registros=10000):
    tipos_processo = ["Ação Trabalhista", "Ação Cível", "Penal"]
    tipos_atendimento = ["Online", "Presencial"]
    status = ["Em Andamento", "Aguardando", "Concluído"]
    status_documentos = ["Completo", "Pendente"]
    
    data = []
    for i in range(num_registros):
        tipo_processo = random.choice(tipos_processo)
        tipo_atendimento = random.choice(tipos_atendimento)
        status_atual = random.choice(status)
        
        tempo_inicio = random.randint(30, 400)
        tempo_atualizacao = random.randint(1, 60)
        
        status_contrato = random.choices(status_documentos, weights=[0.3, 0.7])[0]
        status_proc = random.choices(status_documentos, weights=[0.4, 0.6])[0]
        status_peticao = random.choices(status_documentos, weights=[0.5, 0.5])[0]
        status_doc_complementar = random.choices(status_documentos, weights=[0.3, 0.7])[0]
        
        tempo_resposta = random.randint(30, 100)

        if status_contrato == "Pendente":
            tempo_resposta += random.randint(20, 50)
        if status_proc == "Pendente":
            tempo_resposta += random.randint(10, 40)
        if status_peticao == "Pendente":
            tempo_resposta += random.randint(5, 30)
        if status_doc_complementar == "Pendente":
            tempo_resposta += random.randint(10, 50)
        
        if tipo_processo == "Ação Trabalhista":
            tempo_resposta += random.randint(20, 40)
        elif tipo_processo == "Ação Cível":
            tempo_resposta += random.randint(10, 30)

        if tipo_atendimento == "Presencial":
            tempo_resposta += random.randint(5, 20)
        
        data.append([i+1, tipo_processo, tipo_atendimento, status_atual, tempo_inicio, tempo_atualizacao,
                    status_contrato, status_proc, status_peticao, status_doc_complementar, tempo_resposta])

    df = pd.DataFrame(data, columns=["ID", "Tipo de Processo", "Tipo de Atendimento", "Status", "Tempo de Início (dias)",
                                    "Tempo de Atualização (dias)", "Status Contrato", "Status Procuração",
                                    "Status Petição Inicial", "Status Documento Complementar", "Tempo de Resposta (dias)"])
    
    df.to_csv('dados_processo.csv', index=False)
    print("✅ Dados gerados e salvos em 'dados_processo.csv'")
    return df

def treinar_modelo(df):
    df_processed = df.copy()
    
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
    
    X = df_processed.drop(columns=['Tempo de Resposta (dias)', 'ID'])
    y = df_processed['Tempo de Resposta (dias)']
    
    scaler = StandardScaler()
    numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)']
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ MAE do modelo XGBoost: {mae:.2f} dias")

    # Save model and preprocessors
    try:
        joblib.dump(model, MODEL_FILE)
        preprocessors = {
            'label_encoders': label_encoders,
            'scaler': scaler,
            'colunas': X.columns.tolist()
        }
        joblib.dump(preprocessors, PREPROCESSORS_FILE)
        print("✅ Modelo e pré-processadores salvos com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar modelo: {str(e)}")
    
    return model, label_encoders, scaler, X.columns.tolist()

def carregar_modelo():
    try:
        model_path = os.path.join(os.getcwd(), MODEL_FILE)
        preprocessors_path = os.path.join(os.getcwd(), PREPROCESSORS_FILE)
        
        print(f"🔍 Procurando modelo em: {model_path}")
        print(f"🔍 Procurando pré-processadores em: {preprocessors_path}")
        
        if os.path.exists(model_path) and os.path.exists(preprocessors_path):
            print("✅ Arquivos encontrados. Carregando...")
            model = joblib.load(model_path)
            preprocessors = joblib.load(preprocessors_path)
            return model, preprocessors['label_encoders'], preprocessors['scaler'], preprocessors['colunas']
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

def inicializar_aplicacao():
    global model, label_encoders, scaler, colunas
    
    print("\n🚀 Inicializando aplicação...")
    
    # Try to load existing model first
    loaded_data = carregar_modelo()
    
    if loaded_data is None:
        print("⚠️ Modelo não encontrado ou inválido. Gerando novos dados e treinando...")
        df = gerar_dados_aleatorios(1000)
        model, label_encoders, scaler, colunas = treinar_modelo(df)
        print("✅ Novo modelo treinado com sucesso!")
    else:
        model, label_encoders, scaler, colunas = loaded_data
        print("✅ Modelo carregado com sucesso!")
    
    print("🛠️ Variáveis do modelo:")
    print(f"- Modelo: {'Carregado' if model is not None else 'Não carregado'}")
    print(f"- Número de colunas: {len(colunas) if colunas else 0}\n")

@app.route('/prever-tempo-resposta', methods=['POST'])
def prever_tempo_resposta():
    global model, label_encoders, scaler, colunas
    
    if model is None:
        return jsonify({"erro": "Modelo não carregado"}), 500
    
    try:
        dados = request.json
        
        if not isinstance(dados, list):
            return jsonify({"erro": "Formato inválido. Esperado lista de registros"}), 400
        
        df_input = pd.DataFrame(dados)
        
        if not all(col in df_input.columns for col in colunas):
            return jsonify({"erro": f"Colunas faltando. Esperadas: {colunas}"}), 400
        
        df_input = df_input[colunas]
        df_processed = df_input.copy()
        
        for col, encoder in label_encoders.items():
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df_processed[col] = encoder.transform(df_processed[col])
        
        numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)']
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
        
        predicoes = model.predict(df_processed)
        resposta = [{"tempo_resposta_dias": float(pred)} for pred in predicoes]
        
        return jsonify(resposta)
    
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/')
def home():
    return "API de Previsão de Tempo de Resposta - Envie POST para /prever-tempo-resposta"

# Initialize the application before running
print("🔄 Preparando a aplicação...")
inicializar_aplicacao()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🌍 Iniciando servidor na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)