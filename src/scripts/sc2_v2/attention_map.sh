python ../../main.py --config=attention_map --env-config=sc2_v2_terran with \
    env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=10 \
    env_args.use_extended_action_masking=True check_model_attention_map=True env_args.action_mask=True test_nepisode=5;