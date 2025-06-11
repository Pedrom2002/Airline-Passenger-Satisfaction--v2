#!/usr/bin/env python3
"""
Script de inicialização para a API de Satisfação de Passageiros Aéreos
Autor: Pedro M.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        logger.info("✅ Todas as dependências principais estão instaladas")
        return True
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}")
        logger.info("Execute: pip install -r requirements.txt")
        return False

def check_model_files():
    """Verifica se os arquivos do modelo existem"""
    model_files = [
        "../models/catboost_model.pkl",
        "../models/preprocessor.pkl",
        "../models/feature_names.json"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️  Arquivos de modelo não encontrados: {missing_files}")
        logger.info("A API funcionará em modo de desenvolvimento com predições mock")
    else:
        logger.info("✅ Todos os arquivos do modelo foram encontrados")
    
    return len(missing_files) == 0

def create_directories():
    """Cria diretórios necessários"""
    directories = ["logs", "models", "data", "configs/backups"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Diretório criado/verificado: {directory}")

def set_environment_variables():
    """Define variáveis de ambiente padrão"""
    env_vars = {
        "API_KEY": "sk-development-key",
        "LOG_LEVEL": "INFO",
        "MODEL_VERSION": "v1.0.1",
        "STREAMLIT_SERVER_PORT": "8501",
        "CACHE_ENABLED": "true"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"🔧 Variável de ambiente definida: {key}={value}")

def start_api(host="0.0.0.0", port=8000, reload=False, workers=1):
    """Inicia a API"""
    logger.info(f"🚀 Iniciando API em {host}:{port}")
    
    cmd = [
        "python", "-m", "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if reload:
        cmd.append("--reload")
        logger.info("🔄 Modo de reload ativado")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("⏹️  API interrompida pelo usuário")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao iniciar API: {e}")

def start_streamlit():
    """Inicia a interface Streamlit"""
    logger.info("🎨 Iniciando interface Streamlit")
    
    cmd = [
        "streamlit", "run",
        "app/main.py",
        "--server.port", os.getenv("STREAMLIT_SERVER_PORT", "8501"),
        "--server.address", "0.0.0.0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("⏹️  Streamlit interrompido pelo usuário")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao iniciar Streamlit: {e}")

def check_docker():
    """Verifica se o Docker está disponível e rodando"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Docker encontrado: {result.stdout.strip()}")
            
            # Verificar se o daemon está rodando
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Docker daemon está rodando")
                return True
            else:
                logger.warning("⚠️  Docker está instalado mas o daemon não está rodando")
                logger.info("💡 Inicie o Docker Desktop e tente novamente")
                return False
        else:
            logger.warning("⚠️  Docker não encontrado")
            return False
    except FileNotFoundError:
        logger.warning("⚠️  Docker não está instalado")
        return False
    except Exception as e:
        logger.error(f"❌ Erro ao verificar Docker: {e}")
        return False

def docker_build():
    """Constrói a imagem Docker"""
    if not check_docker():
        logger.error("❌ Docker não está disponível. Use 'python start.py api' para executar sem Docker")
        return False
        
    logger.info("🐳 Construindo imagem Docker")
    
    cmd = ["docker", "build", "-t", "airline-satisfaction-api", "."]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Imagem Docker construída com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao construir imagem Docker: {e}")
        logger.info("💡 Verifique se o Docker Desktop está rodando")
        return False

def docker_run():
    """Executa o container Docker"""
    if not check_docker():
        logger.error("❌ Docker não está disponível")
        return False
        
    logger.info("🐳 Executando container Docker")
    
    # Verificar se a imagem existe
    try:
        result = subprocess.run(["docker", "images", "-q", "airline-satisfaction-api"], 
                               capture_output=True, text=True)
        if not result.stdout.strip():
            logger.warning("⚠️  Imagem não encontrada. Construindo...")
            if not docker_build():
                return False
    except Exception as e:
        logger.error(f"❌ Erro ao verificar imagem: {e}")
        return False
    
    cmd = [
        "docker", "run",
        "-p", "8000:8000",
        "-v", f"{os.getcwd()}/models:/app/models",
        "-v", f"{os.getcwd()}/logs:/app/logs",
        "--env-file", ".env" if os.path.exists(".env") else "/dev/null",
        "airline-satisfaction-api"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        logger.info("⏹️  Container Docker interrompido pelo usuário")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao executar container Docker: {e}")
        return False

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Airline Satisfaction API Launcher")
    parser.add_argument("command", choices=["api", "streamlit", "docker-build", "docker-run", "setup"],
                       help="Comando a executar")
    parser.add_argument("--host", default="0.0.0.0", help="Host para a API")
    parser.add_argument("--port", type=int, default=8000, help="Porta para a API")
    parser.add_argument("--reload", action="store_true", help="Ativar reload automático")
    parser.add_argument("--workers", type=int, default=1, help="Número de workers")
    
    args = parser.parse_args()
    
    # Setup inicial
    if args.command == "setup":
        logger.info("🔧 Configurando ambiente...")
        create_directories()
        set_environment_variables()
        
        if not check_dependencies():
            logger.error("❌ Execute: pip install -r requirements.txt")
            sys.exit(1)
        
        check_model_files()
        logger.info("✅ Setup concluído!")
        logger.info("💡 Execute 'python start.py api' para iniciar a API")
        return
    
    # Verificações básicas
    create_directories()
    set_environment_variables()
    
    if args.command == "api":
        if not check_dependencies():
            logger.error("❌ Execute: pip install -r requirements.txt")
            sys.exit(1)
        check_model_files()
        start_api(args.host, args.port, args.reload, args.workers)
    
    elif args.command == "streamlit":
        if not check_dependencies():
            logger.error("❌ Execute: pip install -r requirements.txt")
            sys.exit(1)
        try:
            import streamlit
            start_streamlit()
        except ImportError:
            logger.error("❌ Streamlit não instalado. Execute: pip install streamlit")
            sys.exit(1)
    
    elif args.command == "docker-build":
        if not docker_build():
            logger.info("💡 Alternativa: Execute 'python start.py api' para rodar sem Docker")
            sys.exit(1)
    
    elif args.command == "docker-run":
        if not docker_run():
            logger.info("💡 Alternativa: Execute 'python start.py api' para rodar sem Docker")
            sys.exit(1)

if __name__ == "__main__":
    main()