function[F]=exm170726(A)
A=input('\n��������ʧЧʱ��:');
M=wblfit(A);%��������A����weibull�ֲ���scale����M��1����shape����M��2��
B=wblrnd(M(1),M(2),[1,10000]);   %����scale����M��1����shape����M��2��ģ��һϵ������

C=sort(B,'ascend');  % ��B�����ݴ�С��������
i=0;j=0;
for k=1:10000
    if C(k)<113  %B10����Ϊ113�����趨������ֵ����·�˵������һ���޸�
        i=i+1;
    end
    if C(k)<96   %B1����Ϊ96�����趨������ֵ����·�˵������һ���޸�
        j=j+1;
    end
end

 fprintf(2,'\n��ģ��10000������\n')
 fprintf(1,'С��B10(113min)������Ϊ��')
  i
  fprintf(1,'\nС��B1(96min)������Ϊ��')
  j
  
  if i>1000
     fprintf(2,'B10ָ�겻�ϸ�\n') 
  else
     fprintf(2,'B10ָ��ϸ�\n')  
  end
  
  if j>100
     fprintf(2,'B1ָ�겻�ϸ�\n') 
  else
     fprintf(2,'B1ָ��ϸ�\n')  
  end
  
clf
plot(C,'r.')
xlabel('����')
ylabel('ʧЧʱ��(min)')
title('����ʧЧʱ�䣨��С��������)')
grid on
hold on
plot([0,10000],[96,96],'-b')
plot([0,10000],[113,113],'-b')
text(-1500,113,'B10(113min)','color','blue')
text(-1500,96,'B1(96min)','color','blue')



    