import gym
from RLmodel import REINFORCE_CNN

if __name__ == "__main__":

    # game = 'procgen-leaper-v0'
    game = 'procgen-coinrun-v0'

    num = 1
    # num = 200

    seed = 123456

    train_env = gym.make(f'procgen:{game}', num_levels=num, start_level=seed, rand_seed=seed, distribution_mode='easy', 
                     use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True)#,render_mode ='human')
    
    test_env = gym.make(f'procgen:{game}', start_level=seed, rand_seed=seed, distribution_mode='easy',
                     use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True)# render_mode='human')

    REINFORCE_CNN_agent = REINFORCE_CNN(train_env)

    # Uncomment to train
    # REINFORCE_CNN_agent.train()

