#!/usr/bin/env bash

comment="my default comment"


if [ -z "$1" ];
then
	echo "No argument supplied"
		else
			comment="$1"
			fi

			pushTo=""


			 
			if [ -z "$2" ];
			 then    
			               echo "No branch supplied"
				                        else
								          pushTo= "origin +"
									  		  pushTo= pushTo + "$2"
											  	  fi


												  echo "value of pushto is" $pushTo

												  git add --all .
												  git commit -m "$comment"
												  git push $pushTo
#git push

