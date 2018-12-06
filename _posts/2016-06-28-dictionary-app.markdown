---
layout: post
title:  "Login Script in Python"
crawlertitle: "Dictionary Manipulation"
summary: "A function that takes password and writes to text  file"
date:   2018-12-06 23:09:47 +0700
categories: posts
tags: 'python'
author: Uzoma
---
Exploring the file handling capabilities in python, I complete the first phase of my word-to-meaning match application in pyton. This phase involves building a script that collects and stores a users-defined  password to a textfile and authenticase user login prior to use of the word-to-meaning macher.
 

See code snippets below:

{% highlight python %}
###### The LOGIN SCRIPT -----> 08/11/2018                 
######  Login() searches for a file password.txt,reads the file, recieves input from user
######  returns TRUE and print welcome message if input exits in password.txt, Else returns FALSE 
     
    
def login():
    aba = glob2.glob("*.txt")
    for i in aba:
        if i == 'password.txt':
            with open ("password.txt") as file:
                holder = file.read().split(",")
                hold = input("Please enter Login Password : ")
                if hold in holder:
                    print("Welcome to the Dictionary App")
                    return True
                else:
                    print("Wrong password")
                    return False



###### newpass() takes arguement TRUE of FALSE, 
#####  if FALSE, it prompts user to create password, and writes or appends password to password.txt file
                    
def newpass(var):
    if var == True:
        pass
    else:
        with open ("password.txt","r+") as newpass:
            contain = input("You MUST CREATE a login password : ")
            lookup = newpass.read().split(",")
            if contain in lookup :
                print("Password exist Already")
            else:
                newpass.write(contain+",")
                print("New Password Saved")
            
{% endhighlight %}

