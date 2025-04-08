.PHONY: pdf ps

all: pdf

ps: ./project-plan.ms
	groff -t -p -K utf8 -m ms -m de -P-pa4 -T ps project-plan.ms > project-plan.ps

pdf: ps
	ps2pdf ./project-plan.ps > project-plan.pdf

