
import yaml
import logging

_log = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            _log.warning("Config file not found at %s. Using default values.", self.config_path)
            return {}
        except (OSError, yaml.YAMLError) as exc:
            _log.warning("Error loading config file %s: %s", self.config_path, exc)
            return {}

    def get(self, section, key, default=None):
        """Gets a configuration value."""
        return self.config.get(section, {}).get(key, default)
