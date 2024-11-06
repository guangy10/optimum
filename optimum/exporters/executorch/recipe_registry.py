recipe_registry = {}


def register_recipe(recipe_name):
    def decorator(func):
        recipe_registry[recipe_name] = func
        return func

    return decorator
