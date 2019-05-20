#ifndef __MYLIST_H__
#define __MYLIST_H__

#include<iostream>
#include<cstdlib>
#include<thread>
#include<fstream>


struct node
{
	std::string word;
	node *next;
};
class list
{
	private:
		node *head;
	public:
		list()
		{
			head=new node;
			head->word="NULL";
			head->next=NULL;
		}
		~list()
		{
			node *__temp;
			while(head->next!=NULL)
			{
				__temp=head;
				head=head->next;
				delete __temp;
			}
			delete head;
		}
		node* getHead()
		{
			return head;
		}
		void DataIn(const char *Filename)
		{
			node *__temp=head;
			std::string TempString;
			std::ifstream fin(Filename);
			if(fin.fail())
			{
				std::cout<<"Data not found."<<std::endl;
				std::system("pause");
				std::exit(0);
			}
			while(!fin.eof())
			{
				std::getline(fin,TempString);
				if(fin.eof())
					break;
				__temp->next=new node;
				__temp=__temp->next;
				__temp->word=TempString;
				__temp->next=NULL;
			}
			fin.close();
		}
		void DataOut(const char *Filename)
		{
			std::ofstream fout(Filename);
			if(fout.fail())
			{
				std::cout<<"Data not found."<<std::endl;
				std::system("pause");
				std::exit(0);
			}
			node *__temp=head;
			while(__temp->next!=NULL)
			{
				__temp=__temp->next;
				fout<<__temp->word<<std::endl;
			}
			fout.close();
			return;
		}
		void PrintList()
		{
			node *__temp=head;
			while(__temp->next!=NULL)
			{
				__temp=__temp->next;
				std::cout<<__temp->word<<std::endl;
			}
			return;
		}
		void Append(std::string &TempString)
		{
			node *__temp=head;
			while(__temp->next!=NULL)
			{
				__temp=__temp->next;
			}
			__temp->next=new node;
			__temp=__temp->next;
			__temp->word=TempString;
			__temp->next=NULL;
			return;
		}
		bool FindString(std::string &TempString)
		{
			node *__temp=head;
			while(__temp->next!=NULL)
			{
				__temp=__temp->next;
				if(TempString==__temp->word)
					return true;
			}
			return false;
		}
		int TellSpace()
		{
			int Cnt=0;
			node *__temp=head;
			while(__temp->next!=NULL)
			{
				__temp=__temp->next;
				Cnt++;
			}
			return Cnt;
		}
		bool isEmpty()
		{
			int Cnt=0;
			node *__temp=head;
			while(__temp->next!=NULL)
			{
				__temp=__temp->next;
				Cnt++;
			}
			return Cnt==0;
		}
};

#endif
