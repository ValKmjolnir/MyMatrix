#ifndef __MYMATRIX_H__
#define __MYMATRIX_H__

#include<iostream>
#include<cmath>

template<typename __T>
class Matrix
{
	private:
		int row;
		int col;
		__T **num;
	public:
		Matrix(const int __row=0,const int __col=0)
		{
			row=__row;
			col=__col;
			if(row!=0&&col!=0)
			{
				num=new __T* [row];
				for(int i=0;i<row;++i)
					num[i]=new __T[col];
			}
			else
				num=NULL;
		}
		Matrix(const Matrix<__T> &__temp)
		{
			row=__temp.row;
			col=__temp.col;
			if(row!=0&&col!=0)
			{
				num=new __T* [row];
				for(int i=0;i<row;++i)
					num[i]=new __T[col];
				for(int i=0;i<row;++i)
					for(int j=0;j<col;++j)
						num[i][j]=__temp.num[i][j];
			}
			else
				num=NULL;
		}
		~Matrix()
		{
			if(num!=NULL)
			{
				for(int i=0;i<row;++i)
					delete[] num[i];
				delete num;
			}
		}
		Matrix operator+(const Matrix<__T> &B)
		{
			if(this->row==B.row&&this->col==B.col)
			{
				for(int i=0;i<row;++i)
					for(int j=0;j<col;++j)
						this->num[i][j]+=B.num[i][j];
				return *this;
			}
			else
			{
				Matrix<__T> NullMatrix(0,0);
				std::string WarningInformation="No matching matrix";
				throw WarningInformation;
				return NullMatrix;
			}
		}
		Matrix operator-(const Matrix<__T> &B)
		{
			if(this->row==B.row&&this->col==B.col)
			{
				for(int i=0;i<row;++i)
					for(int j=0;j<col;++j)
						this->num[i][j]-=B.num[i][j];
				return *this;
			}
			else
			{
				Matrix<__T> NullMatrix(0,0);
				std::string WarningInformation="No matching matrix";
				throw WarningInformation;
				return NullMatrix;
			}
		}
		Matrix operator*(const Matrix<__T> &B)
		{
			if(this->row==0||this->col==0||B.row==0||B.col==0)
			{
				Matrix<__T> NullMatrix(0,0);
				std::string WarningInformation="No matching matrix";
				throw WarningInformation;
				return NullMatrix;
			}
			else if(this->col!=B.row)
			{
				Matrix<__T> NullMatrix(0,0);
				std::string WarningInformation="No matching matrix";
				throw WarningInformation;
				return NullMatrix;
			}
			else
			{
				Matrix<__T> Temp(this->row,B.col);
				__T trans;
				for(int i=0;i<Temp.row;++i)
					for(int j=0;j<Temp.col;++j)
					{
						trans=0;
						for(int k=0;k<this->col;++k)
							trans+=this->num[i][k]*B.num[k][j];
						Temp.num[i][j]=trans;
					}
				return Temp;
			}
		}
		Matrix &operator=(const Matrix<__T> &B)
		{
			if(num!=NULL)
			{
				for(int i=0;i<row;++i)
					delete[] num[i];
				delete num;
			}
			row=B.row;
			col=B.col;
			if(row!=0&&col!=0)
			{
				num=new __T* [row];
				for(int i=0;i<row;++i)
					num[i]=new __T[col];
				for(int i=0;i<row;++i)
					for(int j=0;j<col;++j)
						num[i][j]=B.num[i][j];
			}
			else
				num=NULL;
			return *this;
		}
		template<typename T>
		friend std::ostream &operator<<(std::ostream &strm,Matrix<T> &B);
		template<typename T>
		friend std::istream &operator>>(std::istream &strm,Matrix<T> &B);
		template<typename T>
		friend Matrix<T> Hadamard(const Matrix<T> &A,const Matrix<T> &B);
		template<typename T>
		friend Matrix<T> Transpose(const Matrix<T> &B);
		
};

template<typename T>
std::ostream &operator<<(std::ostream &strm,Matrix<T> &B)
{
	for(int i=0;i<B.row;++i)
	{
		strm<<"|";
		for(int j=0;j<B.col;++j)
		{
			if(j==B.col-1)
				strm<<B.num[i][j];
			else
				strm<<B.num[i][j]<<" ";
		}
		strm<<"|"<<std::endl;
	}
	return strm;
}

template<typename T>
std::istream &operator>>(std::istream &strm,Matrix<T> &B)
{
	for(int i=0;i<B.row;++i)
		for(int j=0;j<B.col;++j)
			strm>>B.num[i][j];
	return strm;
}

template<typename T>
Matrix<T> Hadamard(const Matrix<T> &A,const Matrix<T> &B)
{
	if(A.row==0||A.col==0||B.row==0||B.col==0)
	{
		Matrix<T> NullMatrix(0,0);
		std::string WarningInformation="No matching matrix";
		throw WarningInformation;
		return NullMatrix;
	}
	else if(A.row!=B.row||A.col!=B.col)
	{
		Matrix<T> NullMatrix(0,0);
		std::string WarningInformation="No matching matrix";
		throw WarningInformation;
		return NullMatrix;
	}
	else
	{
		Matrix<T> Temp(A.row,A.col);
		for(int i=0;i<A.row;++i)
			for(int j=0;j<A.col;++j)
				Temp.num[i][j]=A.num[i][j]*B.num[i][j];
		return Temp;
	}
}

template<typename T>
Matrix<T> Transpose(const Matrix<T> &B)
{
	Matrix<T> temp(B.col,B.row);
	for(int i=0;i<B.row;++i)
		for(int j=0;j<B.col;++j)
			temp.num[j][i]=B.num[i][j];
	return temp;
}
#endif
