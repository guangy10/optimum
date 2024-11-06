task_registry = {}


def register_task(recipe_name):
    def decorator(func):
        task_registry[recipe_name] = func
        return func

    return decorator
