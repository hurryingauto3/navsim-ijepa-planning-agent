export PYTHONPATH=/navsim_workspace
# Navtest dataset
python navsim/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=navtest \
    agent=ijepa_agent \
    agent.mlp_weights_path="/navsim_workspace/code/checkpoints/planning_head_20250423_184215_loss0_3079.pth" \
    experiment_name=ijepa_agent_navtest \
    worker=ray_distributed \
    output_dir="code/outputs/ijepa_agent_navtest_results"
    
    
# Test dataset
python navsim/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=test \
    agent=ijepa_agent \
    agent.mlp_weights_path="/navsim_workspace/code/checkpoints/planning_head_20250423_184215_loss0_3079.pth" \
    traffic_agents=non_reactive \
    experiment_name=ijepa_agent_test_nonreactive \
    output_dir="code/outputs/ijepa_agent_navtest_nonreactive_results" \
    worker=ray_distributed

    # Assuming running from /navsim_workspace/
python navsim/navsim/planning/script/run_metric_caching.py \
    +scenario_builder=pdm_test \
    +train_test_split.data_path="dataset/test_navsim_logs/test" \
    train_test_split=test \

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=training_ijepa_planning_agent \
    trainer.params.max_epochs=50 \
    train_test_split=navtrain

# Assuming NAVSIM_DEVKIT_ROOT is set correctly
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=training_ijepa_planning_agent \
    trainer.params.max_epochs=50 \
    train_test_split=navtrain \
    # You might not need the explicit agent_override=... if you changed the default in the main config
    # If you kept the default agent: ego_status_mlp_agent and want to override from command line:
    # agent_override=ijepa_planning_agent # This syntax uses the override group if defined


TRAIN_TEST_SPLIT=navtest
MLP_WEIGHTS=/path/to/your/trained/mlp_weights.pth
EXPERIMENT_NAME=ijepa_planning_agent_eval_one_stage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=ijepa_planning_agent \
    agent.mlp_weights_path=$MLP_WEIGHTS \
    experiment_name=$EXPERIMENT_NAME \
    # --------------------------
    # --- CHANGE THESE LINES ---
    # traffic_agents_policy=non_reactive # Keep this if needed
    # Point this to the .pth file containing your trained MLP state_dict