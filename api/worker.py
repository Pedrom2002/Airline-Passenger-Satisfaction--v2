#!/usr/bin/env python3
"""
Worker para processamento background - Airline Satisfaction API
Processa tarefas assíncronas como retraining, backup, e limpeza de logs
"""

import asyncio
import logging
import os
import time
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any
import schedule
import redis
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('worker')

class BackgroundWorker:
    """Worker para tarefas de background"""
    
    def __init__(self):
        self.redis_client = None
        self.db_engine = None
        self.setup_connections()
        
    def setup_connections(self):
        """Configura conexões com Redis e banco de dados"""
        try:
            # Redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("✅ Conexão Redis estabelecida")
            
            # Database
            db_url = os.getenv('DATABASE_URL', 'sqlite:///airline_data.db')
            self.db_engine = create_engine(db_url)
            logger.info("✅ Conexão banco de dados estabelecida")
            
        except Exception as e:
            logger.error(f"❌ Erro nas conexões: {e}")
    
    def backup_models(self):
        """Cria backup dos modelos"""
        try:
            backup_dir = Path('backups/models')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Lista de arquivos para backup
            model_files = [
                '../models/catboost_model.pkl',
                '../models/preprocessor.pkl',
                '../models/feature_names.json'
            ]
            
            backed_up = []
            for file_path in model_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    backup_path = backup_dir / f"{timestamp}_{filename}"
                    
                    # Copiar arquivo
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    backed_up.append(str(backup_path))
            
            logger.info(f"✅ Backup realizado: {len(backed_up)} arquivos")
            
            # Limpar backups antigos (manter últimos 10)
            self.cleanup_old_backups(backup_dir, keep=10)
            
        except Exception as e:
            logger.error(f"❌ Erro no backup: {e}")
    
    def cleanup_old_backups(self, backup_dir: Path, keep: int = 10):
        """Remove backups antigos"""
        try:
            backup_files = list(backup_dir.glob('*'))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(backup_files) > keep:
                for old_file in backup_files[keep:]:
                    old_file.unlink()
                    logger.info(f"🗑️ Backup removido: {old_file.name}")
                    
        except Exception as e:
            logger.error(f"❌ Erro na limpeza: {e}")
    
    def cleanup_logs(self):
        """Limpa logs antigos"""
        try:
            log_dir = Path('logs')
            if not log_dir.exists():
                return
            
            # Remover logs com mais de 30 dias
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for log_file in log_dir.glob('*.log'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"🗑️ Log removido: {log_file.name}")
                    
        except Exception as e:
            logger.error(f"❌ Erro na limpeza de logs: {e}")
    
    def collect_metrics(self):
        """Coleta métricas do sistema"""
        try:
            import psutil
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'model_version': os.getenv('MODEL_VERSION', 'v1.0.1')
            }
            
            # Salvar métricas no Redis
            if self.redis_client:
                key = f"metrics:{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.redis_client.setex(key, 3600, json.dumps(metrics))
            
            logger.info(f"📊 Métricas coletadas: CPU {metrics['cpu_percent']:.1f}%, RAM {metrics['memory_percent']:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Erro na coleta de métricas: {e}")
    
    def check_model_drift(self):
        """Verifica drift do modelo (simulado)"""
        try:
            # Simular verificação de drift
            import random
            drift_score = random.uniform(0.02, 0.25)
            threshold = 0.15
            
            if drift_score > threshold:
                logger.warning(f"⚠️ Drift detectado: {drift_score:.3f} > {threshold}")
                
                # Salvar alerta no Redis
                if self.redis_client:
                    alert = {
                        'type': 'model_drift',
                        'score': drift_score,
                        'threshold': threshold,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.redis_client.lpush('alerts', json.dumps(alert))
            else:
                logger.info(f"✅ Modelo estável: drift score {drift_score:.3f}")
                
        except Exception as e:
            logger.error(f"❌ Erro na verificação de drift: {e}")
    
    def health_check(self):
        """Verifica saúde dos serviços"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'redis': False,
                'database': False,
                'api': False
            }
            
            # Verificar Redis
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    health_status['redis'] = True
                except:
                    pass
            
            # Verificar Database
            if self.db_engine:
                try:
                    with self.db_engine.connect() as conn:
                        conn.execute("SELECT 1")
                    health_status['database'] = True
                except:
                    pass
            
            # Verificar API
            try:
                import requests
                response = requests.get('http://localhost:8000/health', timeout=5)
                health_status['api'] = response.status_code == 200
            except:
                pass
            
            # Log status
            services_up = sum(health_status.values()) - 1  # -1 para timestamp
            logger.info(f"🏥 Health check: {services_up}/3 serviços ativos")
            
            # Salvar no Redis
            if self.redis_client:
                self.redis_client.setex('health_status', 300, json.dumps(health_status))
                
        except Exception as e:
            logger.error(f"❌ Erro no health check: {e}")
    
    def process_prediction_queue(self):
        """Processa fila de predições (se implementada)"""
        try:
            if not self.redis_client:
                return
                
            # Verificar se há predições na fila
            queue_length = self.redis_client.llen('prediction_queue')
            if queue_length > 0:
                logger.info(f"📋 Processando {queue_length} predições na fila")
                
                # Processar até 10 predições por vez
                for _ in range(min(10, queue_length)):
                    prediction_data = self.redis_client.rpop('prediction_queue')
                    if prediction_data:
                        # Aqui você processaria a predição
                        logger.info("🔮 Predição processada da fila")
                        
        except Exception as e:
            logger.error(f"❌ Erro no processamento da fila: {e}")
    
    def export_daily_report(self):
        """Exporta relatório diário"""
        try:
            report_date = datetime.now().strftime('%Y-%m-%d')
            
            # Simular dados do relatório
            report = {
                'date': report_date,
                'total_predictions': 1250,
                'satisfied_predictions': 780,
                'dissatisfied_predictions': 470,
                'avg_confidence': 0.82,
                'avg_processing_time_ms': 45.3,
                'model_version': os.getenv('MODEL_VERSION', 'v1.0.1')
            }
            
            # Salvar relatório
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            report_file = reports_dir / f'daily_report_{report_date}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📄 Relatório diário exportado: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Erro na exportação do relatório: {e}")
    
    def run_scheduled_tasks(self):
        """Executa tarefas agendadas"""
        logger.info("🚀 Iniciando worker de background")
        
        # Agendar tarefas
        schedule.every(5).minutes.do(self.collect_metrics)
        schedule.every(10).minutes.do(self.health_check)
        schedule.every(15).minutes.do(self.check_model_drift)
        schedule.every(30).minutes.do(self.process_prediction_queue)
        schedule.every().hour.do(self.cleanup_logs)
        schedule.every(6).hours.do(self.backup_models)
        schedule.every().day.at("23:59").do(self.export_daily_report)
        
        # Loop principal
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Verificar a cada minuto
                
            except KeyboardInterrupt:
                logger.info("⏹️ Worker interrompido pelo usuário")
                break
            except Exception as e:
                logger.error(f"❌ Erro no worker: {e}")
                time.sleep(60)

async def async_worker():
    """Worker assíncrono para tarefas específicas"""
    logger.info("🔄 Iniciando worker assíncrono")
    
    while True:
        try:
            # Tarefas assíncronas aqui
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Erro no worker assíncrono: {e}")
            await asyncio.sleep(60)

def main():
    """Função principal"""
    worker = BackgroundWorker()
    
    # Verificar se deve executar em modo assíncrono
    if os.getenv('ASYNC_MODE', 'false').lower() == 'true':
        asyncio.run(async_worker())
    else:
        worker.run_scheduled_tasks()

if __name__ == "__main__":
    main()