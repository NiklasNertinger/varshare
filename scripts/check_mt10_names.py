import metaworld

try:
    mt10 = metaworld.MT10()
    print("MT10 Task Names:")
    for name in mt10.train_classes.keys():
        print(f"- {name}")
except Exception as e:
    print(f"Error loading MT10: {e}")
