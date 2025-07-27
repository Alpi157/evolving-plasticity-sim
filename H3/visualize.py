import pygame, config
from world import World
from creature import Forager
WALL=(60,60,60);FOOD=(0,180,0);AGENT=(0,120,255);BLACK=(0,0,0)
TILE=8;FPS=30;STEPS=300
def draw(window,world,agents):
    window.fill(BLACK)
    for y in range(world.height):
        for x in range(world.width):
            if world.grid[y,x]==1:
                pygame.draw.rect(window,WALL,(x*TILE,y*TILE,TILE,TILE))
    for fx,fy in world.food:
        pygame.draw.circle(window,FOOD,(fx*TILE+TILE//2,fy*TILE+TILE//2),TILE//4)
    for a in agents:
        pygame.draw.circle(window,AGENT,(a.x*TILE+TILE//2,a.y*TILE+TILE//2),TILE//3)
    pygame.display.flip()
class DummyLearner:
    def learn(self,*args,**kwargs): pass
def run_demo(population,world):
    pygame.init()
    win=pygame.display.set_mode((world.width*TILE,world.height*TILE))
    clock=pygame.time.Clock()
    wcopy=world.clone()
    demo_agents=[Forager(a.x,a.y,genome=a.genome.copy()) for a in population]
    for _ in range(STEPS):
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                pygame.quit();return
        for agent in demo_agents:
            agent.step(wcopy,learning_module=DummyLearner())
        draw(win,wcopy,demo_agents)
        clock.tick(FPS)
    pygame.quit()
