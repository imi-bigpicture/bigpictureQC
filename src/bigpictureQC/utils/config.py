import yaml

class Config:
    """Configuration manager to load and access settings from a YAML file."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
    
    def get(self, key, default=None):
        """Retrieve values from nested dictionary using a path-like key."""
        parts = key.split('.')
        current_element = self.settings
        for part in parts:
            if current_element.get(part) is None:
                return default
            current_element = current_element[part]
        return current_element
    
    def __getattr__(self, item):
        """Allows attributes to be accessed directly if they exist."""
        return self.get(item)