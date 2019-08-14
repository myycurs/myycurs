function[F]=exm170726(A)
A=input('\n输入样本失效时间:');
M=wblfit(A);%根据输入A估计weibull分布的scale参数M（1）和shape参数M（2）
B=wblrnd(M(1),M(2),[1,10000]);   %根据scale参数M（1）和shape参数M（2）模拟一系列数据

C=sort(B,'ascend');  % 将B中数据从小到大排列
i=0;j=0;
for k=1:10000
    if C(k)<113  %B10设置为113，如需定义其他值请和下方说明文字一起修改
        i=i+1;
    end
    if C(k)<96   %B1设置为96，如需定义其他值请和下方说明文字一起修改
        j=j+1;
    end
end

 fprintf(2,'\n共模拟10000个样本\n')
 fprintf(1,'小于B10(113min)样本数为：')
  i
  fprintf(1,'\n小于B1(96min)样本数为：')
  j
  
  if i>1000
     fprintf(2,'B10指标不合格\n') 
  else
     fprintf(2,'B10指标合格\n')  
  end
  
  if j>100
     fprintf(2,'B1指标不合格\n') 
  else
     fprintf(2,'B1指标合格\n')  
  end
  
clf
plot(C,'r.')
xlabel('样本')
ylabel('失效时间(min)')
title('样本失效时间（从小到大排列)')
grid on
hold on
plot([0,10000],[96,96],'-b')
plot([0,10000],[113,113],'-b')
text(-1500,113,'B10(113min)','color','blue')
text(-1500,96,'B1(96min)','color','blue')



    