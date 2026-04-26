import structlog
import dagshub
from mlflow.tracking import MlflowClient

logger = structlog.get_logger()

class ChampionPromoter:
    """Human-in-the-loop manual approval script."""
    
    def __init__(self, tracking_repo: str):
        repo_owner, repo_name = tracking_repo.split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        self.client = MlflowClient()
        self.model_name = "StoreCast_XGBoost"

    def approve_candidate(self):
        """Swaps the @candidate tag to @production."""
        try:
            # 1. Get the current candidate
            candidate = self.client.get_model_version_by_alias(self.model_name, "candidate")
            version = candidate.version
            
            logger.info(f"Reviewing {self.model_name} Version {version}...")
            
            # 2. Ask for explicit human approval
            approval = input(f"Approve Version {version} for PRODUCTION? (yes/no): ")
            
            if approval.lower().strip() == 'yes':
                # 3. Promote it
                self.client.set_registered_model_alias(self.model_name, "production", version)
                logger.info(f"🚀 SUCCESS: Version {version} is now LIVE in Production!")
            else:
                logger.warning("Promotion aborted by user.")
                
        except Exception as e:
            logger.error("Could not find a staged @candidate model.", error=str(e))

if __name__ == '__main__':
    from src.utils.config_manager import ConfigManager
    cfg = ConfigManager()
    promoter = ChampionPromoter(tracking_repo=cfg.get("project.tracking_repo"))
    promoter.approve_candidate()