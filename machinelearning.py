import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    
    # Adicionar novas features
    df['Documentos_Pendentes'] = (
        (df['Status Contrato'] == 'Pendente').astype(int) +
        (df['Status Procuração'] == 'Pendente').astype(int) +
        (df['Status Petição Inicial'] == 'Pendente').astype(int) +
        (df['Status Documento Complementar'] == 'Pendente').astype(int)
    )
    
    df['Interacao_Tipo_Status'] = df['Tipo de Processo'] + '_' + df['Status']

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
        'Status Documento Complementar',
        'Interacao_Tipo_Status'  # Nova feature categórica
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    # Features numéricas adicionais
    numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)', 'Documentos_Pendentes']
    
    X = df_processed.drop(columns=['Tempo de Resposta (dias)', 'ID'])
    y = df_processed['Tempo de Resposta (dias)']
    
    # Transformação logarítmica da variável alvo
    USE_LOG_TARGET = True
    if USE_LOG_TARGET:
        y = np.log1p(y)
        print("Aplicada transformação logarítmica na variável alvo")

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Modelo com hiperparâmetros otimizados
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.8,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mae'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Avaliação
    y_pred = model.predict(X_test)
    
    # Reverter transformação logarítmica se aplicada
    if USE_LOG_TARGET:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("Avaliação do Modelo")
    print(f"MAE: {mae:.2f} dias")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} dias")
    print(f"R²: {r2:.2f}")
    print("="*50 + "\n")

    # Gráfico de dispersão melhorado
    plt.figure(figsize=(12, 10))
    
    # Criar DataFrame para plotagem
    results = pd.DataFrame({
        'Real': y_test,
        'Previsto': y_pred
    })
    
    # Calcular linha de tendência
    slope, intercept, r_value, p_value, std_err = stats.linregress(results['Real'], results['Previsto'])
    line = slope * results['Real'] + intercept
    
    # Plot principal com seaborn
    ax = sns.jointplot(
        x='Real',
        y='Previsto',
        data=results,
        kind='reg',
        joint_kws={
            'line_kws': {'color': 'red', 'label': f'Linha de Tendência (y={slope:.2f}x + {intercept:.2f})'},
            'scatter_kws': {'alpha': 0.3, 's': 15}
        },
        marginal_kws={'bins': 30, 'color': 'skyblue'}
    )
    
    # Linha de 45 graus
    min_val = min(results.min())
    max_val = max(results.max())
    ax.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Linha de 45°')
    
    # Adicionar métricas
    metrics_text = (
        f'R² = {r2:.2f}\n'
        f'MAE = {mae:.2f} dias\n'
        f'RMSE = {rmse:.2f} dias\n'
        f'Inclinação = {slope:.2f}'
    )
    
    ax.ax_joint.text(
        0.05, 0.95,
        metrics_text,
        transform=ax.ax_joint.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.suptitle('Valor Real vs. Previsto - Tempo de Resposta', y=1.02)
    ax.ax_joint.legend(loc='lower right')
    ax.ax_joint.set_xlabel('Valor Real (dias)', fontsize=12)
    ax.ax_joint.set_ylabel('Valor Previsto (dias)', fontsize=12)
    ax.ax_joint.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('real_vs_previsto_melhorado.png', dpi=300, bbox_inches='tight')
    print("Gráfico de validação salvo como 'real_vs_previsto_melhorado.png'")
    plt.close()

    # Validação cruzada
    try:
        # Preparar dados para cross-validation
        if USE_LOG_TARGET:
            y_cv = np.log1p(df_processed['Tempo de Resposta (dias)'])
        else:
            y_cv = df_processed['Tempo de Resposta (dias)']
            
        cv_scores = cross_val_score(
            model, X, y_cv,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(f"RMSE Validação Cruzada: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f} dias")
    except Exception as e:
        print(f"Erro na validação cruzada: {e}")

    return model, label_encoders, scaler, X.columns.tolist(), USE_LOG_TARGET

def carregar_modelo_ou_treinar_novo():
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSORS_FILE):
            print("Carregando modelo e pré-processadores existentes...")
            model = joblib.load(MODEL_FILE)
            preprocessors = joblib.load(PREPROCESSORS_FILE)
            label_encoders = preprocessors['label_encoders']
            scaler = preprocessors['scaler']
            colunas = preprocessors['colunas']
            use_log_target = preprocessors['use_log_target']
            print("Modelo e pré-processadores carregados com sucesso!")
            return model, label_encoders, scaler, colunas, use_log_target
    except Exception as e:
        print(f"Erro ao carregar modelo existente: {e}")
        print("Treinando novo modelo...")

    df = gerar_dados_aleatorios(50000)
    model, label_encoders, scaler, colunas, use_log_target = treinar_modelo(df)

    try:
        print("Salvando modelo e pré-processadores...")
        joblib.dump(model, MODEL_FILE)
        preprocessors = {
            'label_encoders': label_encoders,
            'scaler': scaler,
            'colunas': colunas,
            'use_log_target': use_log_target
        }
        joblib.dump(preprocessors, PREPROCESSORS_FILE)
        print("Modelo e pré-processadores salvos com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar modelo: {e}")

    return model, label_encoders, scaler, colunas, use_log_target

app = Flask(__name__)

model = None
label_encoders = None
scaler = None
colunas = None
use_log_target = None

@app.route('/prever-tempo-resposta', methods=['POST'])
def prever_tempo_resposta():
    global model, label_encoders, scaler, colunas, use_log_target

    if model is None:
        return jsonify({"erro": "Modelo não carregado"}), 500

    try:
        dados = request.json

        if not isinstance(dados, list):
            return jsonify({"erro": "Formato inválido. Esperado lista de registros"}), 400

        df_input = pd.DataFrame(dados)

        # Adicionar novas features necessárias
        if 'Documentos_Pendentes' not in df_input.columns:
            df_input['Documentos_Pendentes'] = (
                (df_input['Status Contrato'] == 'Pendente').astype(int) +
                (df_input['Status Procuração'] == 'Pendente').astype(int) +
                (df_input['Status Petição Inicial'] == 'Pendente').astype(int) +
                (df_input['Status Documento Complementar'] == 'Pendente').astype(int)
            )
        
        if 'Interacao_Tipo_Status' not in df_input.columns:
            df_input['Interacao_Tipo_Status'] = df_input['Tipo de Processo'] + '_' + df_input['Status']

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
        numeric_cols = ['Tempo de Início (dias)', 'Tempo de Atualização (dias)', 'Documentos_Pendentes']
        if numeric_cols:
            df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

        # Previsão
        predicoes = model.predict(df_processed)
        
        # Reverter transformação logarítmica se aplicada
        if use_log_target:
            predicoes = np.expm1(predicoes)
        
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
    global model, label_encoders, scaler, colunas, use_log_target

    print("Inicializando modelo...")
    model, label_encoders, scaler, colunas, use_log_target = carregar_modelo_ou_treinar_novo()
    print("Modelo inicializado e API pronta para receber requisições")

if __name__ == '__main__':
    # Inicializa a aplicação
    inicializar_aplicacao()
    port = 5000

    try:
        # Configuração do Ngrok
        NGROK_AUTHTOKEN = "2y9do1E9kJTxn0iMEkIU08NIZG9_uKsJHf3Hnazk4qzdDAgU"  # Substitua pelo seu token real
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