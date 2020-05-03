/*By ValKmjolnir 2020/5/3                        */
/*Before using this header file                  */
/*Make sure that you have added iostream & cmath */
#ifndef __MYMATRIX_H__
#define __MYMATRIX_H__

#include<iostream>

template<typename __T>
class Matrix
{
private:
	int row;
	int col;
	__T **num;
public:
	Matrix(const int,const int);
	Matrix(const Matrix<__T>&);
	~Matrix();
	Matrix  operator+ (const Matrix<__T>&);
	Matrix  operator- (const Matrix<__T>&);
	Matrix  operator* (const Matrix<__T>&);
	Matrix& operator= (const Matrix<__T>&);
	__T*    operator[](const int);
	Matrix  Hadamard  (const Matrix<__T>&,const Matrix<__T>&);
	Matrix  Transpose (const Matrix<__T>&);
	template<typename T> friend std::ostream& operator<<(std::ostream&,const Matrix<T>&);
	template<typename T> friend std::istream& operator>>(std::istream&,const Matrix<T>&);
};

template<typename __T>
Matrix<__T>::Matrix(const int __row,const int __col)
{
	row=__row;
	col=__col;
	if(row>0 && col>0)
	{
		num=new __T* [row];
		for(int i=0;i<row;++i)
			num[i]=new __T[col];
	}
	else
	{
		row=0;
		col=0;
		num=NULL;
	}
	return;
}

template<typename __T>
Matrix<__T>::Matrix(const Matrix<__T>& __temp)
{
	row=__temp.row;
	col=__temp.col;
	if(row>0 && col>0)
	{
		num=new __T* [row];
		for(int i=0;i<row;++i)
			num[i]=new __T[col];
		for(int i=0;i<row;++i)
			for(int j=0;j<col;++j)
				num[i][j]=__temp.num[i][j];
	}
	else
	{
		row=0;
		col=0;
		num=NULL;
	}
	return;
}

template<typename __T>
Matrix<__T>::~Matrix()
{
	if(num)
	{
		for(int i=0;i<row;++i)
			delete[] num[i];
		delete num;
	}
	return;
}

template<typename __T>
Matrix<__T> Matrix<__T>::operator+(const Matrix<__T>& B)
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

template<typename __T>
Matrix<__T> Matrix<__T>::operator-(const Matrix<__T>& B)
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

template<typename __T>
Matrix<__T> Matrix<__T>::operator*(const Matrix<__T>& B)
{
	Matrix<__T> NullMatrix(0,0);
	if(this->row==0||this->col==0||B.row==0||B.col==0)
	{
		std::string WarningInformation="No matching matrix";
		throw WarningInformation;
	}
	else if(this->col!=B.row)
	{
		std::string WarningInformation="No matching matrix";
		throw WarningInformation;
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
	return NullMatrix;
}

template<typename __T>
Matrix<__T>& Matrix<__T>::operator=(const Matrix<__T>& B)
{
	if(num)
	{
		for(int i=0;i<row;++i)
			delete[] num[i];
		delete num;
	}
	row=B.row;
	col=B.col;
	if(row>0 && col>0)
	{
		num=new __T* [row];
		for(int i=0;i<row;++i)
			num[i]=new __T[col];
		for(int i=0;i<row;++i)
			for(int j=0;j<col;++j)
				num[i][j]=B.num[i][j];
	}
	else
	{
		row=0;
		col=0;
		num=NULL;
	}
	return *this;
}

template<typename __T>
__T* Matrix<__T>::operator[](const int addr)
{
	return addr>=this->row? NULL:this->num[addr];
}

template<typename __T>
Matrix<__T> Matrix<__T>::Hadamard(const Matrix<__T>& A,const Matrix<__T>& B)
{
	Matrix<__T> NullMatrix(0,0);
	if(!A.row || !A.col || !B.row || !B.col)
	{
		std::string WarningInformation="No matching matrix";
		throw WarningInformation;
	}
	else if(A.row!=B.row||A.col!=B.col)
	{
		std::string WarningInformation="No matching matrix";
		throw WarningInformation;
	}
	else
	{
		Matrix<__T> Temp(A.row,A.col);
		for(int i=0;i<A.row;++i)
			for(int j=0;j<A.col;++j)
				Temp.num[i][j]=A.num[i][j]*B.num[i][j];
		return Temp;
	}
	return NullMatrix;
}

template<typename __T>
Matrix<__T> Matrix<__T>::Transpose(const Matrix<__T>& B)
{
	Matrix<__T> temp(B.col,B.row);
	for(int i=0;i<B.row;++i)
		for(int j=0;j<B.col;++j)
			temp.num[j][i]=B.num[i][j];
	return temp;
}

template<typename T>
std::ostream& operator<<(std::ostream& strm,const Matrix<T>& aim)
{
	for(int i=0;i<aim.row;++i)
	{
		for(int j=0;j<aim.col;++j)
			strm<<aim.num[i][j]<<((char)(j==aim.col-1)? '\n':' ');
	}
	return strm;
}

template<typename T>
std::istream& operator>>(std::istream& strm,const Matrix<T>& aim)
{
	for(int i=0;i<aim.row;++i)
		for(int j=0;j<aim.col;++j)
			strm>>aim.num[i][j];
	return strm;
}
#endif
