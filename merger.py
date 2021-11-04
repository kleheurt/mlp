f2data = "" 
with open('.\\t2.json') as f2: 
  f2data = '\n' + f2.read()
    
with open('.\\t1.json','a+') as f1:
    f1.write(f2data)