from .latent_tracker import LatentTracker

NODE_CLASS_MAPPINGS = {
    "latent_tracker": LatentTracker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "latent_tracker": "Latent Value Tracker"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']