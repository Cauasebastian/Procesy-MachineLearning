import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, jsonify
from pyngrok import ngrok

MODEL_FILE = 'modelo_tempo_resposta.pkl'
PREPROCESSORS_FILE = 'preprocessors.pkl'

# Configurar seeds para reprodutibilidade
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def gerar_dados_aleatorios(num_registros=50000):
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

        # Base mais realista com interações entre features
        base_value = random.gauss(80, 20)
        process_bonus = 20 if tipo_processo == "Ação Trabalhista" else 0
        attendance_bonus = 15 if tipo_atendimento == "Presencial" else 0
        contract_penalty = 40 if status_contrato == "Pendente" else 0
        proc_penalty = 30 if status_proc == "Pendente" else 0
        petition_penalty = 20 if status_peticao == "Pendente" else 0
        doc_penalty = 35 if status_doc_complementar == "Pendente" else 0
        
        tempo_resposta = base_value + process_bonus + attendance_bonus + contract_penalty + proc_penalty + petition_penalty + doc_penalty
        tempo_resposta = max(30, min(tempo_resposta + random.uniform(-15, 15), 150))

        data.append([i+1, tipo_processo, tipo_atendimento, status_atual, tempo_inicio, tempo_atualizacao,
                    status_contrato, status_proc, status_peticao, status_doc_complementar, tempo_resposta])

    df = pd.DataFrame(data, columns=["ID", "Tipo de Processo", "Tipo de Atendimento", "Status", "Tempo de Início (dias)",
                                    "Tempo de Atualização (dias)", "Status Contrato", "Status Procuração",
                                    "Status Petição Inicial", "Status Documento Complementar", "Tempo de Resposta (dias)"])

    df.to_csv('dados_processo.csv', index=False)
    print(f"Dados gerados ({num_registros} registros) e salvos em 'dados_processo.csv'")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Modelo com hiperparâmetros otimizados (sem early stopping)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print("Avaliação do Modelo")
    print(f"MAE: {mae:.2f} dias")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} dias")
    print("="*50 + "\n")

    # Validação cruzada
    try:
        cv_scores = cross_val_score(
            model, X, y, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(f"RMSE Validação Cruzada: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f} dias")
    except Exception as e:
        print(f"Erro na validação cruzada: {e}")

    return model, label_encoders, scaler, X.columns.tolist()

def carregar_modelo_ou_treinar_novo():
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSORS_FILE):
            print("Carregando modelo e pré-processadores existentes...")
            model = joblib.load(MODEL_FILE)
            preprocessors = joblib.load(PREPROCESSORS_FILE)
            label_encoders = preprocessors['label_encoders']
            scaler = preprocessors['scaler']
            colunas = preprocessors['colunas']
            print("Modelo e pré-processadores carregados com sucesso!")
            return model, label_encoders, scaler, colunas
    except Exception as e:
        print(f"Erro ao carregar modelo existente: {e}")
        print("Treinando novo modelo...")

    df = gerar_dados_aleatorios(50000)
    model, label_encoders, scaler, colunas = treinar_modelo(df)

    try:
        print("Salvando modelo e pré-processadores...")
        joblib.dump(model, MODEL_FILE)
        preprocessors = {
            'label_encoders': label_encoders,
            'scaler': scaler,
            'colunas': colunas
        }
        joblib.dump(preprocessors, PREPROCESSORS_FILE)
        print("Modelo e pré-processadores salvos com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar modelo: {e}")

    return model, label_encoders, scaler, colunas

app = Flask(__name__)

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
        dados = request.json

        if not isinstance(dados, list):
            return jsonify({"erro": "Formato inválido. Esperado lista de registros"}), 400

        df_input = pd.DataFrame(dados)

        # Verificar colunas necessárias
        colunas_necessarias = colunas.copy()
        if 'ID' in colunas_necessarias:
            colunas_necessarias.remove('ID')
            
        if not all(col in df_input.columns for col in colunas_necessarias):
            return jsonify({"erro": f"Colunas faltando. Esperadas: {colunas_necessarias}"}), 400

        df_processed = df_input[colunas_necessarias].copy()

        # Pré-processamento
        for col, encoder in label_encoders.items():
            # Tratar categorias não vistas
            novas_categorias = set(df_processed[col]) - set(encoder.classes_)
            if novas_categorias:
                print(f"Aviso: Novas categorias encontradas em {col}: {novas_categorias}")
            # Mapear para categoria existente ou default
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df_processed[col] = encoder.transform(df_processed[col])

        # Escalonamento
        numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)']
        if numeric_cols:
            df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

        # Previsão
        predicoes = model.predict(df_processed)
        resposta = [{"tempo_resposta_dias": float(pred)} for pred in predicoes]

        return jsonify(resposta)

    except Exception as e:
        import traceback
        print(f"Erro: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"erro": "Erro interno no processamento"}), 500

@app.route('/')
def home():
    return "API de Previsão de Tempo de Resposta - Envie POST para /prever-tempo-resposta"

def inicializar_aplicacao():
    global model, label_encoders, scaler, colunas

    print("Inicializando modelo...")
    model, label_encoders, scaler, colunas = carregar_modelo_ou_treinar_novo()
    print("Modelo inicializado e API pronta para receber requisições")

if __name__ == '__main__':
    # Inicializa a aplicação
    inicializar_aplicacao()
    port = 5000
    
    try:
        # Configuração do Ngrok - MODO SEGURO (leia as observações abaixo)
        NGROK_AUTHTOKEN = "2y9do1E9kJTxn0iMEkIU08NIZG9_uKsJHf3Hnazk4qzdDAgU"  # Substitua pelo seu token real
        
        # Método recomendado (seguro):
        # 1. Opção preferida: Variável de ambiente (execute no terminal antes de rodar o script)
        # export NGROK_AUTHTOKEN="seu_token_aqui" (Linux/Mac)
        # set NGROK_AUTHTOKEN="seu_token_aqui" (Windows)
        
        # Ou 2. Leia de um arquivo de configuração externo (mais seguro que hardcoded)
        ngrok.set_auth_token(os.getenv('NGROK_AUTHTOKEN', NGROK_AUTHTOKEN))
        
        # Conecta o túnel Ngrok
        public_url = ngrok.connect(port).public_url
        print(f"\n{'='*50}")
        print(f" * Ngrok tunnel: {public_url} -> http://127.0.0.1:{port}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\n{'='*50}")
        print("Erro na configuração do Ngrok:")
        print(f"Detalhes: {str(e)}")
        print("A aplicação continuará em modo local apenas")
        print(f" * Servidor local: http://127.0.0.1:{port}")
        print(f"{'='*50}\n")
        public_url = None
    
    # Inicia a aplicação Flask
    app.run(port=port)