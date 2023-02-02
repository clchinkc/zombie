import time,sys,os,threading,copy,random
from getkey import getkey,keys
from mazes import master_maze
from replit import db,clear
# defineses delay print which prints charitors one at a time
def delay_print(s,speed=0.05):
	global stop
	if speed==0.05 and speedrun:speed=0
	for c in s:
		sys.stdout.write(c)
		sys.stdout.flush()
		time.sleep(speed)
		if stop:
			stop=True
			break

money=0
roomx=0
roomy=3
xpos=2
ypos=2
xblit=0
yblit=0
max_health=3
health=3
runery=10
style=0
oil=0
bombs=0
cp=[0,3,2,2]
boss=False
push=0
darkness=False
doubleCost=False
debug=True
dir=0
past=""
sword=False
non=False
char=0
name=str(os.environ['REPL_OWNER'])
stop=False
hotkeys={}
# definces the diffnt responces to naming yourself diffent things
name_responses={
	# "Slick":["That's a pretty slick name you got there son","\nI have a friend who likes to call himself that"],
	"EvanMessina":["I have a good friend who goes by that name, son","\nhe likes to call himself slick"],
	# "Robert":["That is going to be a bad choice, son"],
	# "Lincoln":["Wow you could have put in a better name, son","\nhow about Richard?"],
	# "Richard":["Genshin impact sucks"],
	"AlexanderGiller":["I know you","\nyou're the one that made me"],
	# "Hoisington":["THATS ME!","\nIt can't be!","\nNot again!"],
	"BryanCheng2":["Nice!"],
	# "Joshua":["Oh him, he owes me quite a bit of homework"],
	# "Ethan":["You suck"],
	# "Julia":["You're his sister arent you?"],
	# "Debug":[colored("I'm not gonna let you in that easy","red"),colored("\nyou're gonna need a passcode","red")],
	# "Bug":[colored("debug mode activated","red")],
	# "Dark":[colored("Very Spooky!",attrs=["dark"])],
	# "Sus":[colored("SUS","red")],
	# "James":["smol boi"],
	# "Lion":["ROAR"]
}
icons=["  ","██","██","💰","▲▲","❤️ ","🤑","▓▓","🪙 ","📦","🔘","✉️ ","🔥","⍓⍓","⍄⍄","⍌⍌","⍃⍃","✉️ ",'  ','👿','██']
speednames=["Speed","Speedrun","A","Zoom"]
speedrun=False
runner_start=0
inventory=['map']
state="talking"
cursor_pos=0
past_rooms=[]
playing=True
saving=False
escape=False
fire_time=0
level_time=float('inf')
fire_speed=2
save_order=('xpos','ypos','roomx','roomy','health','oil','bombs','max_health','past_rooms','save_maze','money','inventory','cp')
startMenu=('Continue','New Game','How to play','Leaderboard','Credits')
HowToPlay="Proceed through the mazes using the arrow keys or wasd.\nFind secrets along the way to get coins, and buy items to \nhelp you on your quest.\nAccess the menus by pressing enter.\nSolve puzzles, push boxes into switches, and trek through \ndarkness.\nAnd don't let Mr. Hoisington catch you on your chromebook!"#write how to play
def box_check():
	boxes=set()
	for y in range(len(room)):
		for x in range(len(room[0])):
			if room[y][x] in range(-8,-4):
				boxes.add((x,y))
	for x in boxes:
		conveyer_box(x[0],x[1])
# hadles what happens if an error accurs
def save():
	global save_order,save_maze
	if not (non or debug):
		save={}
		save_maze=copy.deepcopy(master_maze)
		for y in save_maze:
			for x in y:
					x.pop(0)
		for x in save_order:
			save[x]=globals()[x]
		db[name]=save
		print('Saved!')
def load():
	global xpos,ypos,roomx,roomy,health,oil,bombs,max_health,past_rooms,master_maze
	SaveFile=db[name]
	for i in SaveFile:
		if i=='save_maze':
			save_maze=SaveFile[i]
			for y in range(len(master_maze)):
				# clear()
				# print('['+('#'*y)+(' '*(len(master_maze)-y)+']'))
				for x in range(len(master_maze[y])):
					master_maze[y][x][1:]=save_maze[y][x]
		globals()[i]=SaveFile[i]
def refresh():
	global runery,saving,level_time,health,fire_time,room,state
	while playing:
		runery-=1
		if state=="maze":
			if level_time<time.time()-150:
				health-=1
				level_time+=10
				if health<=0:
					respawn()
			if fire_time<time.time() and escape:
				new_fire=set()
				for y in range(len(room)):
					for x in range(len(room[0])):
						if room[y][x]in(12,-9):
							if y<len(room)-1 and room[y+1][x]in(0,2):new_fire.add((x,y+1))
							if y>0 and room[y-1][x]in(0,2):new_fire.add((x,y-1))
							if x<len(room[0])-1 and room[y][x+1]in(0,2):new_fire.add((x+1,y))
							if x>0 and room[y][x-1]in(0,2):new_fire.add((x-1,y))
				for x in new_fire:
					room[x[1]][x[0]]=-9
				if room[ypos][xpos]==-9:
					health-=1
					if health<=0:
						#death
						state='dying'
						print(draw_maze(death=True))
						time.sleep(1)
						delay_print("Oof that's gotta burn")
						time.sleep(0.5)
						respawn()
						time.sleep(1)
						state='maze'
					else:print(draw_maze("It burns!"))
				fire_time+=fire_speed
			conveyor()
			box_check()
			display=draw_maze("PASSIVE"*debug)
			if display:print(display)
		if roomx==2 and roomy==1:time.sleep(0.1)
		elif escape:time.sleep(fire_speed)
		else:time.sleep(0.5)
PassiveUp=threading.Thread(target=refresh)
def on_error():
	global roomx,roomy,xpos,ypos,cp
	print("An error accured, SUS")
	if speedrun:print(time.time() -start)
	respawn()
# controls the actions the your key preses cause while in the maze
def respawn():
	global xpos,ypos,roomx,roomy,health,maze,room,level_time
	roomx,roomy,xpos,ypos=cp
	health=3
	level_time=float('inf')
	maze=master_maze[roomy][roomx]
	room=maze[0].get_maze()
	for y in range(len(room)):
		for x in range(len(room[0])):
			if room[y][x] in(-2,-9):room[y][x]=0
	print(draw_maze("RIP"))
def leaderboard():
	if non:return'Sorry you need an account to see the leaderboard'
	lb=db['lb']
	display=""
	for x in lb:
		if not x:display+='-None-\n'
		else:display+=f'{x[0]}: {int(x[1]//60)}:{int(x[1]%60)}.{round(x[1]%1*100)}\n'
	return display
getLB=threading.Thread(target=leaderboard)
def on_press(key):
	global xblit,yblit,state,dir,past,char,fire_speed
	if key in [keys.UP,"w"]:
		dir=0
		yblit-=1
	elif key in [keys.DOWN,"s"]:
		dir=2
		yblit+=1
	elif key in [keys.LEFT,"a"]:
		dir=3
		xblit-=1
	elif key in [keys.RIGHT,"d"]:
		dir=1
		xblit+=1
	elif key in ('1','2','3','4'):
		state='menu'
		past=None
		hotkey(key)
		print(draw_maze())
	if key=='m':
		state="menu"
		past=None
		map()
		print(draw_maze())
	# elif key == "q":attack()
	# elif key == 's':save()
	elif key == keys.ENTER:
		past=None
		state="menu"
		print(menu())
	else:movement()
	code=[keys.UP,keys.UP,keys.DOWN,keys.DOWN,keys.LEFT,keys.RIGHT,keys.LEFT,keys.RIGHT,'b','a']
	if key==code[char] and escape:
		char+=1
		if char==10:
			delay_print('\u001b[31mHE HE HE HA\u001b[0m',0.1)
			fire_speed/=2
			char=0
def conveyor():
	global xpos,ypos
	xtest=xpos
	ytest=ypos
	dir=room[ypos][xpos]-13
	if dir not in range(4):dir = room[ypos][xpos]+8
	if dir not in range(4):return
	if dir==0:ytest-=1
	elif dir==1:xtest+=1
	elif dir==2:ytest+=1
	elif dir==3:xtest-=1
	if room[ytest][xtest] in [0,13,14,15,16,-8,-7,-6,-5]:
		xpos=xtest
		ypos=ytest
def conveyer_box(xpos,ypos):
	global room
	dest=[]
	room[ypos][xpos]+=21
	if (room[ypos][xpos]-13)==0:
		dest=[ypos-1,xpos]
	elif (room[ypos][xpos]-13)==1:
		dest=[ypos,xpos+1]
	elif (room[ypos][xpos]-13)==2:
		dest=[ypos+1,xpos]
	elif (room[ypos][xpos]-13)==3:
		dest=[ypos,xpos-1]
	if room[dest[0]][dest[1]] in range(13,17):room[dest[0]][dest[1]]-=21
	elif room[dest[0]][dest[1]] ==10:
		room[dest[0]][dest[1]]=-4
		complete=True
		for y in room:
			for x in y:
				if x ==10:complete=False
		for y in range(len(room)):   
			for x in range(len(room[0])):
				if room[y][x] in (4,-1) and complete:
					room[y][x]=0
	elif room[dest[0]][dest[1]]==0: room[dest[0]][dest[1]]=-2
	else:room[ypos][xpos]-=21
# controls the shop
# def attack():
# 	global xpos,ypos,sword
# 	cross=[xpos,ypos]
# 	if dir==0:cross[1]-=1
# 	elif dir==1:cross[0]+=1
# 	elif dir==2:cross[1]+=1
# 	elif dir==3:cross[0]-=1
# 	if room[cross[1]][cross[0]]==0:sword=True
# 	else:room[cross[1]][cross[0]]=0
def hotkey(key):
	global hotkeys,cursor_pos
	if key in hotkeys.keys():return use(hotkeys[key])
	else:
		cursor_pos=0
		while True:
			message=''
			for x in range(len(inventory)):
				if x==cursor_pos:message+="➞ "
				else:message+="  "
				message+= inventory[x].capitalize()
				if x=="lantern":message+="   "+str(oil)
				elif x=="bombs":message+="   "+str(bombs)
				message+="\n"
			if cursor_pos==len(inventory):message+="➞ Cancel"
			else:message+="  Cancel"
			clear()
			print(message)
			i=getkey()
			if i in ('w',keys.UP) and cursor_pos>0:cursor_pos-=1
			elif i in (keys.DOWN,'s') and cursor_pos<len(inventory):cursor_pos+=1
			elif i in('d',keys.ENTER,keys.RIGHT):
				if cursor_pos<len(inventory):
					hotkeys[key]=inventory[cursor_pos]
				return
def main_menu():
	global cursor_pos,debug,isSave
	if non:isSave=False
	else:isSave=name in db.keys()
	char=0
	key='None'
	if not isSave:cursor_pos+=1
		
	while True:
		message=''
		for x in range(len(startMenu)):
			if not isSave and x==0:continue
			if cursor_pos==x:message+="➞ "
			else:message+="  "
			message+=startMenu[x]+"\n"
		message+='\n'
		if cursor_pos==2:message+=HowToPlay
		elif cursor_pos==3:message+=leaderboard()
		clear()
		sys.stdout.write(message)
		key=getkey()
		if key in (keys.DOWN,'s') and cursor_pos<len(startMenu)-1:cursor_pos+=1
		elif key in (keys.UP,'w') and cursor_pos>1-isSave:cursor_pos-=1
		elif key in (keys.RIGHT,keys.ENTER,'d'):
			if cursor_pos==0:return
			elif cursor_pos==1:
				if isSave:
					del db[name]
					isSave=False
				return
			elif cursor_pos==4:
				f=open('credits.txt')
				clear()
				for line in f:
					sys.stdout.write(line)
					time.sleep(0.5)
				sys.stdout.flush()
				f.close()
				time.sleep(1)
		if not non and key==os.environ['debug'][char]:
			char+=1
			if char==len(os.environ['debug']):
				delay_print('\u001b[31mdebug mode activated')
				debug=True
				char=0
				key='None'
		else:char=0
			
def use(item):
	global inventory,past_rooms,roomx,roomy,xpos,ypos,style,oil,song,darkness,bombs,doubleCost,room,maze,max_health,health
	if item=='map':map()
	elif item=="goggles":delay_print("Looking through these goggles you can see he trueth")
	elif item=="heart":
		max_health+=1
		health=max_health
	elif item=="dark":darkness=not darkness
	elif item=="double cost":doubleCost=not doubleCost
	elif item=="up":
		roomy-=1
	elif item=="right":roomx+=1
	elif item=="down":roomy+=1
	elif item=="left":roomx-=1
	elif item=="refresh":
		maze=master_maze[roomy][roomx]
		room=maze[0].get_maze()
	elif item=="random":
		roomx=17
		xpos=ypos=roomy=2
		maze=master_maze[roomy][roomx]
		room=maze[0].get_maze()
	elif item=="village":
		xpos,ypos,roomx,roomy=7,7,9,3
		maze=master_maze[roomy][roomx]
		room=maze[0].get_maze()
	elif item=="rope" and len(past_rooms)>0:
		roomx,roomy,xpos,ypos=past_rooms.pop()
		maze=master_maze[roomy][roomx]
		room=maze[0].get_maze()
	elif item=="bombs":
		e=False
		for x in range(xpos-1,xpos+2):
			if x>=0 and x<len(room[0]):
				for y in range(ypos-1,ypos+2):
					if (y>=0 and y<len(room)) and (room[y][x]in(2,7,20)):e=True
		if e:
			bombs-=1
			room=master_maze[roomy][roomx][0].get_maze()
			for x in range(xpos-1,xpos+2):
				if x>=0 and x<len(room[0]):
					for y in range(ypos-1,ypos+2):
						if (y>=0 and y<len(room)) and room[y][x]in(7,2,20):
							room[y][x]=0
			if bombs<=0: inventory.remove("bombs")
			return draw_maze("BOOM!")
		else:delay_print("There is nothing to blow up!")
			
	elif item=="nuke":
		for x in range(xpos-2,xpos+3):
			if x>=1 and x<len(room[0])-1:
				for y in range(ypos-2,ypos+3):
					if (y>=1 and y<len(room)-1) or ((y>=0 and y<len(room)-1) and (room[y][x]==7 or room[y][x]==2)):room[y][x]=0
		return draw_maze("BOOM!")
	elif "lantern"==item:
		e=False
		for y in room:
			for x in y:
				if x==2:e=True
		if 'dark'in maze or e:
			if oil>0:	
				if "dark" in maze:maze.remove("dark")
				else:
					for y in range(len(room)):
						for x in range(len(room[0])):
							if room[y][x]==2:
								room[y][x]=0
				oil-=1
			else:
				inventory.remove("lantern")
				return draw_maze("You don't have enough oil!")
		else:return delay_print("There is nothing to brighten")
	elif item=="shades":
		time.sleep(0.5)
		if style<=5:delay_print("You feel very slick in these")
		elif style<15:delay_print("You feel as if this joke is going to old soon")
		elif style==15:delay_print("You know what your gonna have to earn this")
		elif style==16:delay_print("You know what how about 100 points")
		elif style==17:delay_print("Then we will see who has the most syle")
		elif style==18:delay_print("You know what I'll even through in a little gift to help you")
		elif style==19:delay_print("It's a style point counter")
		elif style==20:delay_print("Do you like it?")
		elif style==21:delay_print("All right I'll leave you alone")
		elif style==90:delay_print("You're almost there!")
		elif style==91:delay_print("Took you long enough")
		elif style==92:delay_print("But I'm gonna have to have to stop you")
		elif style==93:delay_print("You can't have what lies ahead")
		elif style==94:delay_print("It if far too powerful for the likes of you")
		elif style==97:delay_print("I have to stop you")
		elif style==98:delay_print("STOP",2)
		elif style==99:
			delay_print(random.choices(list('abcdefghijklmnopqrstuvwxyz'),k=10000),0.005)
			# prints the wiki artical
			# f=open("style.txt")
			# delay_print(f.read(),0.008)
			# f.close()
		elif style==100:
			delay_print("Wow I can't belive that you actually did it wow!")
			time.sleep(0.5)
			delay_print("\nwell I supose you deserve this")
			time.sleep(0.5)
			delay_print("\nyou got the nuke")
			style+=1
			# gives the nuke to the player
			inventory.append("nuke")
			return draw_maze()
		time.sleep(0.5)
		delay_print("\nYou got 1 style point!")
		style+=1
		time.sleep(1)
	return draw_maze()
def run_shop(shop,key):
	global money,inventory,oil,cursor_pos,bombs
	if key == keys.ESC:return draw_maze()
	if key in [keys.ENTER,"d",keys.RIGHT]:
		if cursor_pos==len(shop):
			cursor_pos=0
			return draw_maze()
		elif shop[cursor_pos][1]<=money and shop[cursor_pos][0] not in inventory:
			money -= shop[cursor_pos][1]
			if shop[cursor_pos][0]=="lamp oil":
				oil+=1
			elif shop[cursor_pos][0]=="3 bombs":
				if bombs>0 and "bombs" not in inventory:inventory.append("bombs")
				bombs+=3
			else:inventory.append(shop[cursor_pos][0])
	clear()
	message="  $"+str(money)+"\n"
	for x in range(len(shop)):
		if x == cursor_pos:message+="➞ "
		else:message+="  "
		if shop[x][0] not in inventory:message += shop[x][0].capitalize() +"   $"+str(shop[x][1])+"\n"
		else:message+="--Sold Out--\n"
	if cursor_pos==len(shop):message+="➞ "
	else:message+="  "
	message+="Exit"
	return message
#handles the menu
def menu(key=None):
	global inventory,cursor_pos
	if oil>0 and "lantern" not in inventory:inventory.append("lantern")
	if bombs>0 and "bombs" not in inventory:inventory.append("bombs")
	if key == keys.ESC: return draw_maze()
	if key in (keys.UP,"w"):
		if cursor_pos>0:cursor_pos-=1
		else:cursor_pos=len(inventory)
	elif key in (keys.DOWN,"s"):
		if cursor_pos<len(inventory):cursor_pos+=1
		else:cursor_pos=0
	elif key in [keys.ENTER,"d",keys.RIGHT]:
		if cursor_pos>=len(inventory):return draw_maze()
		else:return use(inventory[cursor_pos])
	clear()
	# adds the bigginging of he menu including health,money,and style
	message="money: $"+str(money)+"\nhealth:"+"❤️ "*health+"\n"
	if style >=20 and style!=101:message+="style: "+str(style)+"\n"
	
	n=0
	i=globals()
	if debug:
		for x in i:
			message+=str(x)+"     "+str(i[x])+"\n"
	for x in inventory:
		if n==cursor_pos:message+="➞ "
		else:message+="  "
		message+= x.capitalize()
		if x=="lantern":message+="   "+str(oil)
		elif x=="bombs":message+="   "+str(bombs)
		message+="\n"
		n+=1
	if cursor_pos==len(inventory):message+="➞ Exit"
	else:message+="  Exit"
	return message
def map():
	global roomx,roomy,room,maze
	cursorx=roomx
	cursory=roomy
	while True:
		message=''
		for y in range(len(master_maze)):
			for x in range(roomx-5,roomx+5):
				if x<0 or x>len(master_maze[y])-1:continue
				if debug:
					if y==cursory and cursorx==x:message+='➞ '
					else:message+='  '
				if master_maze[y][x][0]==None:message+=' '*10
				elif 'visit' in master_maze[y][x] or debug:message+=master_maze[y][x][0].name+' '*(10-len(master_maze[y][x][0].name))
				else:message+=' ???????? '
			message+='\n'
		if not debug:message+='Press Enter to Leave'
		clear()
		print(message)
		if debug:
			key=getkey()
			if key in (keys.UP,'w'):cursory-=1
			elif key in (keys.DOWN,'s'):cursory+=1
			elif key in (keys.LEFT,'a'):cursorx-=1
			elif key in (keys.RIGHT,'d'):cursorx+=1
			elif key==keys.ENTER:
				roomx,roomy=cursorx,cursory
				maze=master_maze[roomy][roomx]
				room=maze[0].get_maze()
				break
		else:
			input()
			break
# takes in the current maze and renders it
def draw_maze(message="",death=False):
	global state,past
	state="maze"
	ylen=len(room)
	xlen=len(room[0])
	display=""
	if level_time<float('inf'):
		TimeLeft=level_time+150-time.time()
		if (TimeLeft<60 and TimeLeft%1>.5)or TimeLeft<10:
			display+='\u001b[31m'
		display+=time.strftime('time:%M:%S\n\u001b[0m',time.gmtime(TimeLeft))
	display+='health:'+('❤️ '*health)+'\n'
	# if dir==0:cross[1]-=1
	# elif dir==1:cross[0]+=1
	# elif dir==2:cross[1]+=1
	# elif dir==3:cross[0]-=1
	dark="dark" in maze or darkness
	for y in range(ylen):
		for x in range(xlen):
			t=room[y][x]
			if death and xpos==x and ypos==y:display+='💀'
			elif xpos==x and ypos==y and (name=="EvanMessina" or "shades"in inventory):display+="\U0001F60E"
			elif xpos==x and ypos==y and style>=100:display+="🕶️ "
			elif xpos==x and ypos==y:display+="\U0001F642"
			# elif room[y][x]==0 and x==cross[0] and y==cross[1] and sword:display+="x "
			# elif room[y][x]==0 and x==cross[0] and y==cross[1]:display+=colored("× ","red")
			elif runery==y and x==2 and 'run' in maze and "shades" not in inventory:display+="🏃"
			elif t==-9 and fire_time<time.time()+1:display+='🔥'
			elif not dark or (x<=xpos+1 and x>=xpos-1 and y<=ypos+1 and y>=ypos-1):
				if t== 3 and "money" in master_maze[roomy][roomx]:display+="  "
				elif t == 5 and "cp" not in master_maze[roomy][roomx]:display+="  "
				elif t == 7 and "bombs" not in inventory:display+='\u2588\u2588'
				elif t==2 and "goggles" in inventory:display+="▒▒"
				elif t in [-1,-3]:display+="  "
				elif t in [-2,-5,-6,-7,-8]:display+="📦"
				elif t==-4:display+="✅"
				elif t==18 and escape:display+="██"
				elif t==20 and escape:display+='  '
				else:display+=icons[room[y][x]]
			elif dark and room[y][x] in (1,2,7) and (x==0 or x==xlen-1 or y==0 or y==ylen-1):display+='░░'
			else:display+="  "
		display+="\n"
	if display!=past:
		clear()
		past=display
		if escape and debug:
			display+=str(fire_time-time.time())
		display+= message
		return display
# handeles player movement including colision and interactions with blocks
def update_pos(xtest,ytest):
	global xpos,ypos,display,sword
	xpos=xtest
	ypos=ytest
	display=draw_maze("ACTIVE"*debug)
	if display:print(display)
	sword=False
def movement():
	global xblit,yblit,xpos,ypos,roomy,roomx,runery,inventory,health,state,past_rooms,cp,money,room,maze,cursor_pos,push,playing,saving,speedrun,level_time,fire_time,escape,fire_speed
	xtest=xpos+xblit
	ytest=ypos+yblit
	ylen=len(room)
	xlen=len(room[0])
	# moves the player between rooms
	if ytest<0 or ytest>ylen-1 or xtest<0 or xtest>xlen-1:
		past_rooms.append([roomx,roomy,xpos,ypos])
		dir = [ytest>ylen-1,xtest<0,ytest<0,xtest>xlen-1].index(True)
		runery=15
		if dir==2:roomy-=1
		elif dir==3:roomx+=1
		elif dir==0:roomy+=1
		elif dir==1:roomx-=1
		maze=master_maze[roomy][roomx]
		if not maze[0]:
			if dir==0:roomy-=1
			elif dir==1:roomx+=1
			elif dir==2:roomy+=1
			elif dir==3:roomx-=1
			maze=master_maze[roomy][roomx]
		elif maze[0].end: #winning
			playing=False
			delay_print(maze[0].end.format(name=name))
			if speedrun and not debug:
				length=time.time()-start
				delay_print(time.strftime(f"\ntime:{int(length//60)}:{(len(str(int(length%60)))==0)*'0'}{int(length%60)}.{int(round((length%1)*100))}"))
				time.sleep(1)
				try:
					lb=db["lb"]
				except TypeError:print('can not post your time as you have not created an account')
				else:
					if not lb[9] or length<lb[9][1]:
						print('in')
						for x in range(10):
							if not lb[x] or length<lb[x][1]:
								i=-1
								lb.insert(x,[name,length])
								for y in range(x+1,10):
									if not lb[y] or lb[y][0]==name:
										i=y
										break
								lb.pop(i)
								break
							elif lb[x][0]==name:break
				print("\n\n"+leaderboard())
			time.sleep(5)
			# if name in db.keys():del db[name]
			f=open('credits.txt')
			clear()
			for line in f:
				sys.stdout.write(line)
				time.sleep(0.5)
			sys.stdout.flush()
			f.close()
			time.sleep(1)
			print('\nCode for playtesters:Yooo!')
			return
		room=maze[0].get_maze()
		exits=maze[0].get_exits()
		if dir in (0,2):
			xtest=exits[dir]
			ytest=(len(room)-1)*(dir//2)
		else:
			ytest=exits[dir]
			if dir == 1:xtest=len(room[0])-1
			else:xtest=0
		complete=False
		if 'done' in maze:complete=True
		for y in range(len(room)):
			for x in range(len(room[0])):
				if room[y][x]==-1:room[y][x]=4
				elif room[y][x] in(-2,-9):room[y][x]=0
				elif room[y][x]==-3 and not complete:room[y][x]=9
				elif room[y][x]==-4 and not complete:room[y][x]=10
				elif room[y][x] in range(-8,-4):room[y][x]+=21
		update_pos(xtest,ytest)
		if roomx==10 and roomy==1:level_time=time.time()
		else:level_time=float('inf')
		if escape:
			room[ypos][xpos]=-9
			fire_time=time.time()+fire_speed*2
		if 'visit' not in maze:maze.append('visit')
	elif room[ytest][xtest]==3 and "money" not in master_maze[roomy][roomx]:
		# Coin bags
		money+=5
		xpos=xtest
		ypos=ytest
		print(draw_maze("You got 5 coins"))
		master_maze[roomy][roomx].append("money")
	elif room[ytest][xtest]==4:
		# Spikes
		health-=1
		if health<=0:
			#death
			xpos,ypos=xtest,ytest
			print(draw_maze(death=True))
			state='dying'
			time.sleep(1)
			delay_print("Oof that's gotta hurt")
			time.sleep(0.5)
			respawn()
			time.sleep(0.1)
		else:
			#hurt
			xpos=xtest
			ypos=ytest
			room[ytest][xtest]=-1
			print(draw_maze("Ouch"))
	elif room[ytest][xtest]==5 and "cp" in maze:
		# checkpoint
		health=max_health
		xpos=xtest
		ypos=ytest
		cp=[roomx,roomy,xpos,ypos]
		save()
		if not isSave and time.time()-start<=60:speedrun=True
		if escape:
			fire_speed/=2
			maze.remove('cp')
		print(draw_maze(),"Health refilled")
	elif room[ytest][xtest]==6:
		# shop
		global shop,past
		past=None
		state="shop"
		shop=master_maze[roomy][roomx][1]
		if doubleCost:
			for x in shop:
				x[1]*=2
		cursor_pos=0
		# delay_print("Lamp oil, rope, bombs you want it, it's your's my friend as long as you have enough coins",0.03)
		print(run_shop(shop,None))
	elif room[ytest][xtest]==8:
		# coin
		money+=1
		xpos=xtest
		ypos=ytest
		print(draw_maze("You got a coin"))
		room[ytest][xtest]=0
	elif room[ytest][xtest] in [-2,9,-5,-6,-7,-8] and (ytest+yblit<0 or ytest+yblit>len(room)-1 or xtest+xblit<0 or xtest+xblit>len(room[0])-1):update_pos(xtest,ytest)
	elif room[ytest][xtest] in [-2,9,-5,-6,-7,-8] and room[ytest+yblit][xtest+xblit] in[0,10,-3,13,14,15,16]:
		# handles BOX
			if room[ytest][xtest]==9:room[ytest][xtest]=-3
			elif room[ytest][xtest] in range(-8,-4):room[ytest][xtest]+=21
			else:room[ytest][xtest]=0
			if room[ytest+yblit][xtest+xblit] == 0:room[ytest+yblit][xtest+xblit]=-2
			elif room[ytest+yblit][xtest+xblit] == -3:room[ytest+yblit][xtest+xblit]=9
			elif room[ytest+yblit][xtest+xblit] in range(13,17):room[ytest+yblit][xtest+xblit]-=21
			else:
				room[ytest+yblit][xtest+xblit]=-4
				complete=True
				for y in room:
						if 10 in y:complete=False
				if complete:
					maze.append('done')
					for y in range(len(room)):   
						for x in range(len(room[0])):
							if room[y][x] in (4,-1):room[y][x]=0
			update_pos(xtest,ytest)
	elif room[ytest][xtest]in[11,17,19]:
		# sign
		if escape and room[ytest][xtest]==11:
			return update_pos(xtest,ytest)
		if(type(maze[1]))!=str:message=maze[2]
		else:message=maze[1]
		state="talking"
		delay_print(message,0.05)
		state="maze"
		if room[ytest][xtest]==17:room[ytest][xtest]=0
		elif room[ytest][xtest]==19:
			health=max_health
			cp=[roomx,roomy,xpos,ypos]
			escape=True
			fire_time=time.time()+5
			maze[1]='He he he ha'
	elif room[ytest][xtest]==12 or (room[ytest][xtest]==-9 and fire_time<time.time()+1):
		health-=1
		if health<=0:
			#death
			print(draw_maze(death=True))
			time.sleep(1)
			delay_print("Oof that's gotta burn")
			time.sleep(0.5)
			respawn()
			time.sleep(1)
		else:
			#hurt
			xpos=xtest
			ypos=ytest
			print(draw_maze("It burns!"))
	elif room[ytest][xtest] in (1,7,9,-2,10) or ((room[ytest][xtest]==20 and not escape)or(room[ytest][xtest]==18 and escape)):
		pass
	elif ytest<=runery and 'run' in maze and "shades" not in inventory:
		state="talking"
		# runner
		time.sleep(0.5)
		delay_print("Wow you caught up to me")
		time.sleep(0.5)
		delay_print("\nYou are very fast")
		time.sleep(0.5)
		delay_print("\nhere take these")
		time.sleep(0.5)
		delay_print("\nyou deserve them")
		time.sleep(1)
		inventory.append("shades")
		xpos=xtest
		ypos=ytest
		print(draw_maze(),"You got the slick shades")
	else:update_pos(xtest,ytest)
	# if xblit>5:xblit=5
	# elif xblit<-5:xblit=-5
	# if yblit>5:yblit=5
	# elif yblit<-5:yblit=-5	
	yblit=0
	xblit=0
try:db['lb']
except:
	print('Get an account you bozo')
	time.sleep(0.05)
	clear()
	non=True
	delay_print("What is your name? ")
	name = input().capitalize()
	time.sleep(0.5)
main_menu()
# start of game
# delay_print("\nHello I am Mr. Hoisington")
# time.sleep(0.5)
# delay_print('\nWelcome to my experience')
# time.sleep(0.5)
if name in name_responses.keys():
	for x in name_responses[name]:
		delay_print(x,0.05)
		time.sleep(.5)
	if name=="Sick":money+=10
	elif name=="Richard":money-=5
	elif name=="Dark":darkness=True
	elif name==os.environ.get('debug'):debug=True
	elif name=="Sus":
		f=open("NOTHING.txt")
		raise Exception("\n"+f.read())
elif name in speednames:
	print("speedrun acivated")
	speedrun=True
else:
  delay_print(f"Welcome {name} to my game")
if not non and name in db.keys():
	load()
# time.sleep(0.5)
# delay_print("\nbut that is a story for another time.")
# time.sleep(0.5)
# delay_print("\nAnyway let's have an experance son")
# time.sleep(0.5)
# delay_print("\nit is a simple maze")
# time.sleep(0.5)
# delay_print("\nwith many secrets in the walls to discover")
# time.sleep(.5)
# delay_print("\nGood Luck")
# time.sleep(1)
cursor_pos=0
maze=master_maze[roomy][roomx]
room=maze[0].get_maze()
if debug:
	money=float('inf')
	bombs=float('inf')
	oil=float('inf')
	# db['lb'][0]=None
	print(db['lb'])
	inventory=["dark","double cost","village","heart","goggles","random",'rope','nuke',"bombs"]
if non or name not in db.keys():start=time.time()
# cp=[roomx,roomy,xpos,ypos]
state="maze"
# game loop
PassiveUp.start()
while playing:
	key = getkey()
	if state=="maze":
		on_press(key)
	elif state == "shop":
		# moves the cursor position in the shop
		if key in (keys.UP,"w"):
			if cursor_pos > 0:cursor_pos-=1
			else:cursor_pos=len(shop)
		elif key in (keys.DOWN,"s"):
			if cursor_pos < len(shop):cursor_pos+=1
			else:cursor_pos=0
		print_shop=run_shop(shop,key)
		if print_shop != None:print(print_shop)
	elif state == "menu":
		# sends key preses and calls run_shop
		print(menu(key))